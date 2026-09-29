"""Evaluate PySteps nowcasting baseline on Cyclone Michaung (Dec 2023).

Computes real CSI, FSS, and Brier Score per lead-time band (0-1h, 1-2h, 2-3h)
against IMD thresholds in configs/thresholds.yaml, compared to Persistence
and Climatology baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xarray as xr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rews.config import get_settings
from rews.nowcasting.baseline_pysteps import (
    PyStepsNowcaster,
    compute_brier_score,
    compute_csi,
    compute_fss,
)
from rews.utils.logging import get_logger

logger = get_logger("rews.scripts.evaluate_nowcasting")


def run_nowcasting_evaluation(
    datacube_path: str = "data/processed/datacube.zarr",
    event_id: str = "chennai_michaung_2023",
    output_report_path: str = "outputs/reports/nowcasting_michaung_evaluation.json",
    n_ens_members: int = 15,
    eval_stride_steps: int = 2,  # Evaluate every 1 hour
) -> Dict[str, Any]:
    settings = get_settings()
    logger.info("Opening datacube from %s...", datacube_path)
    ds = xr.open_zarr(datacube_path)

    time_series = pd.to_datetime(ds.time.values, utc=True)
    precip_all = ds.precipitation_rate.values  # (T, H, W)

    # Locate event window
    matching_events = [ev for ev in settings.pipeline.splits.get("test_events", []) if ev["id"] == event_id]
    if not matching_events:
        raise ValueError(f"Event {event_id} not found in test_events configuration")

    ev_info = matching_events[0]
    t_start = pd.to_datetime(ev_info["start_utc"])
    t_end = pd.to_datetime(ev_info["end_utc"])

    mask = (time_series >= t_start) & (time_series <= t_end)
    event_indices = np.where(mask)[0]
    logger.info("Found %d steps for %s (%s to %s)", len(event_indices), ev_info["name"], t_start.isoformat(), t_end.isoformat())

    # Pre-calculate domain climatology from training partitions
    train_masks = []
    for tr_ev in settings.pipeline.splits.get("train_events", []):
        tr_s = pd.to_datetime(tr_ev["start_utc"])
        tr_e = pd.to_datetime(tr_ev["end_utc"])
        train_masks.append((time_series >= tr_s) & (time_series <= tr_e))
    all_train_mask = np.logical_or.reduce(train_masks)
    train_precip = precip_all[all_train_mask]
    climatology_mean = float(np.mean(train_precip))
    climatology_field = np.full((precip_all.shape[1], precip_all.shape[2]), climatology_mean, dtype=np.float32)

    # Thresholds from config
    csi_thresholds = [0.5, 2.5, 5.0, 10.0, 15.6]
    fss_thresholds = [2.5, 10.0]
    fss_scales_km = [10, 20, 40, 80]
    pixel_scales = {10: 1, 20: 2, 40: 4, 80: 8}  # ~10 km per 0.1 deg cell

    # Lead-time bands:
    # 0-1h: lead 1, 2 (+30m, +60m)
    # 1-2h: lead 3, 4 (+90m, +120m)
    # 2-3h: lead 5, 6 (+150m, +180m)
    bands = {
        "0-1h": [0, 1],
        "1-2h": [2, 3],
        "2-3h": [4, 5],
    }

    nowcaster = PyStepsNowcaster(settings=settings, n_cascade_levels=3)

    # Evaluation loop
    lead_steps = 6
    context_steps = 3

    # Accumulators for metrics
    results_by_band = {
        band_name: {
            "csi_sprog": {th: [] for th in csi_thresholds},
            "csi_steps_mean": {th: [] for th in csi_thresholds},
            "csi_persistence": {th: [] for th in csi_thresholds},
            "csi_climatology": {th: [] for th in csi_thresholds},
            "fss_sprog": {th: {km: [] for km in fss_scales_km} for th in fss_thresholds},
            "fss_persistence": {th: {km: [] for km in fss_scales_km} for th in fss_thresholds},
            "brier_steps": {th: [] for th in csi_thresholds},
            "brier_persistence": {th: [] for th in csi_thresholds},
            "brier_climatology": {th: [] for th in csi_thresholds},
        }
        for band_name in bands
    }

    valid_eval_cycles = 0
    t0 = time.time()

    # Iterate over event steps with stride
    for i in range(context_steps, len(event_indices) - lead_steps, eval_stride_steps):
        curr_idx = event_indices[i]
        curr_time = time_series[curr_idx]

        # Context frames: t-2, t-1, t
        context = precip_all[curr_idx - (context_steps - 1) : curr_idx + 1]
        # Ground truth observations: t+1 to t+6
        obs = precip_all[curr_idx + 1 : curr_idx + 1 + lead_steps]

        # Only evaluate during periods with active precipitation in domain
        if np.nanmax(context[-1]) < 0.1 and np.nanmax(obs) < 0.1:
            continue

        valid_eval_cycles += 1

        # 1. Estimate motion
        velocity = nowcaster.estimate_motion(context)

        # 2. S-PROG deterministic forecast
        fc_sprog = nowcaster.predict_deterministic(context, lead_steps=lead_steps, method="sprog", velocity=velocity)

        # 3. STEPS probabilistic forecast
        fc_steps = nowcaster.predict_probabilistic(
            context,
            lead_steps=lead_steps,
            n_ens_members=n_ens_members,
            seed=42 + i,
            velocity=velocity,
        )
        fc_steps_mean = np.mean(fc_steps, axis=0)

        # 4. Persistence forecast
        fc_pers = nowcaster.predict_persistence(context, lead_steps=lead_steps)

        # 5. Climatology forecast
        fc_clim = np.repeat(climatology_field[np.newaxis, :, :], lead_steps, axis=0)

        # Evaluate across bands
        for band_name, step_indices in bands.items():
            for s_idx in step_indices:
                o_frame = obs[s_idx]
                sprog_frame = fc_sprog[s_idx]
                steps_mean_frame = fc_steps_mean[s_idx]
                pers_frame = fc_pers[s_idx]
                clim_frame = fc_clim[s_idx]

                # CSI
                for th in csi_thresholds:
                    results_by_band[band_name]["csi_sprog"][th].append(compute_csi(sprog_frame, o_frame, th))
                    results_by_band[band_name]["csi_steps_mean"][th].append(compute_csi(steps_mean_frame, o_frame, th))
                    results_by_band[band_name]["csi_persistence"][th].append(compute_csi(pers_frame, o_frame, th))
                    results_by_band[band_name]["csi_climatology"][th].append(compute_csi(clim_frame, o_frame, th))

                # FSS
                for th in fss_thresholds:
                    for km in fss_scales_km:
                        scale_px = pixel_scales[km]
                        results_by_band[band_name]["fss_sprog"][th][km].append(compute_fss(sprog_frame, o_frame, th, scale_px))
                        results_by_band[band_name]["fss_persistence"][th][km].append(compute_fss(pers_frame, o_frame, th, scale_px))

                # Brier Score (STEPS exceedance probability)
                for th in csi_thresholds:
                    prob_steps = np.mean((fc_steps[:, s_idx, :, :] >= th).astype(np.float32), axis=0)
                    prob_pers = (pers_frame >= th).astype(np.float32)
                    prob_clim = float(climatology_mean >= th)
                    prob_clim_field = np.full_like(prob_steps, prob_clim)

                    results_by_band[band_name]["brier_steps"][th].append(compute_brier_score(prob_steps, o_frame, th))
                    results_by_band[band_name]["brier_persistence"][th].append(compute_brier_score(prob_pers, o_frame, th))
                    results_by_band[band_name]["brier_climatology"][th].append(compute_brier_score(prob_clim_field, o_frame, th))

        if valid_eval_cycles % 10 == 0:
            logger.info("Evaluated %d forecast origins on %s (%.1fs elapsed)...", valid_eval_cycles, curr_time.strftime("%Y-%m-%d %H:%M"), time.time() - t0)

    logger.info("Evaluation complete! Evaluated %d total active forecast cycles.", valid_eval_cycles)

    # Aggregate metric summaries
    summary = {
        "event_id": event_id,
        "event_name": ev_info["name"],
        "total_evaluated_cycles": valid_eval_cycles,
        "climatology_mean_mm_h": climatology_mean,
        "bands": {},
    }

    print("\n" + "=" * 100)
    print(f"STAGE 3 PYSTEPS NOWCASTING BENCHMARK REPORT: {ev_info['name']}")
    print(f"Total Evaluated Forecast Cycles: {valid_eval_cycles} | Stride: {eval_stride_steps*30} min | Climatology Mean: {climatology_mean:.3f} mm/h")
    print("=" * 100)

    for band_name in bands:
        b_data = results_by_band[band_name]
        summary["bands"][band_name] = {"csi": {}, "fss": {}, "brier": {}}

        print(f"\n>>> LEAD-TIME BAND: {band_name}")
        print("-" * 100)
        print(f"{'Threshold (mm/h)':<18} | {'PySteps S-PROG':<15} | {'PySteps STEPS':<15} | {'Persistence':<15} | {'Climatology':<15}")
        print("-" * 100)

        for th in csi_thresholds:
            csi_sprog = float(np.mean(b_data["csi_sprog"][th]))
            csi_steps = float(np.mean(b_data["csi_steps_mean"][th]))
            csi_pers = float(np.mean(b_data["csi_persistence"][th]))
            csi_clim = float(np.mean(b_data["csi_climatology"][th]))

            summary["bands"][band_name]["csi"][str(th)] = {
                "sprog": csi_sprog,
                "steps_mean": csi_steps,
                "persistence": csi_pers,
                "climatology": csi_clim,
            }
            print(f"{th:<18.1f} | {csi_sprog:<15.4f} | {csi_steps:<15.4f} | {csi_pers:<15.4f} | {csi_clim:<15.4f}")

        # FSS
        print("-" * 100)
        print("FRACTIONS SKILL SCORE (FSS):")
        print(f"{'Threshold':<12} | {'Scale':<10} | {'S-PROG FSS':<15} | {'Persistence FSS':<18} | {'Gain over Pers':<15}")
        print("-" * 100)
        for th in fss_thresholds:
            summary["bands"][band_name]["fss"][str(th)] = {}
            for km in fss_scales_km:
                fss_sprog = float(np.mean(b_data["fss_sprog"][th][km]))
                fss_pers = float(np.mean(b_data["fss_persistence"][th][km]))
                gain = fss_sprog - fss_pers
                summary["bands"][band_name]["fss"][str(th)][f"{km}km"] = {
                    "sprog": fss_sprog,
                    "persistence": fss_pers,
                    "gain": gain,
                }
                print(f"{th:<12.1f} | {f'{km} km':<10} | {fss_sprog:<15.4f} | {fss_pers:<18.4f} | {gain:<+15.4f}")

        # Brier Score & BSS
        print("-" * 100)
        print("BRIER SCORE & BRIER SKILL SCORE (BSS):")
        print(f"{'Threshold':<12} | {'STEPS BS':<15} | {'Persistence BS':<18} | {'Climatology BS':<18} | {'BSS (vs Clim)':<15}")
        print("-" * 100)
        for th in csi_thresholds:
            bs_steps = float(np.mean(b_data["brier_steps"][th]))
            bs_pers = float(np.mean(b_data["brier_persistence"][th]))
            bs_clim = float(np.mean(b_data["brier_climatology"][th]))
            bss_clim = 1.0 - (bs_steps / bs_clim) if bs_clim > 0 else 0.0

            summary["bands"][band_name]["brier"][str(th)] = {
                "brier_steps": bs_steps,
                "brier_persistence": bs_pers,
                "brier_climatology": bs_clim,
                "bss_vs_climatology": bss_clim,
            }
            print(f"{th:<12.1f} | {bs_steps:<15.5f} | {bs_pers:<18.5f} | {bs_clim:<18.5f} | {bss_clim:<+15.4f}")

    print("=" * 100)

    # Save to JSON report
    out_p = Path(output_report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved nowcasting benchmark report to %s", out_p)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PySteps nowcaster on Cyclone Michaung")
    parser.add_argument("--datacube", type=str, default="data/processed/datacube.zarr")
    parser.add_argument("--event", type=str, default="chennai_michaung_2023")
    parser.add_argument("--output", type=str, default="outputs/reports/nowcasting_michaung_evaluation.json")
    parser.add_argument("--ens", type=int, default=15)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    run_nowcasting_evaluation(
        datacube_path=args.datacube,
        event_id=args.event,
        output_report_path=args.output,
        n_ens_members=args.ens,
        eval_stride_steps=args.stride,
    )
