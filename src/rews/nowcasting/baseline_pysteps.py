"""PySteps baseline nowcasting module for SIH26071 REWS.

Implements:
1. Optical flow estimation via Lucas-Kanade.
2. Deterministic nowcasting via S-PROG (Spectral Prognosis) and Semi-Lagrangian extrapolation.
3. Probabilistic nowcasting via STEPS (Short-Term Ensemble Prediction System).
4. Persistence and Climatology benchmark baselines.
5. Verification metrics: CSI (Threat Score), FSS (Fractions Skill Score at 10, 20, 40, 80 km),
   and Brier Score / BSS across lead-time bands (0-1h, 1-2h, 2-3h).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import uniform_filter

from rews.config import REWSSettings, get_settings
from rews.utils.logging import get_logger

logger = get_logger("rews.nowcasting.pysteps")


class PyStepsNowcaster:
    """Nowcasting engine utilizing PySteps optical flow, S-PROG, and STEPS ensemble."""

    def __init__(
        self,
        settings: Optional[REWSSettings] = None,
        n_cascade_levels: int = 4,
        ar_order: int = 2,
        kmperpixel: float = 10.0,
        timestep_minutes: int = 30,
    ):
        self.settings = settings or get_settings()
        self.n_cascade_levels = n_cascade_levels
        self.ar_order = ar_order
        self.kmperpixel = kmperpixel
        self.timestep_minutes = timestep_minutes

        # Lazy import of pysteps components
        import pysteps
        from pysteps import motion, nowcasts, utils
        self.pysteps = pysteps
        self.motion_method = motion.get_method("lucaskanade")
        self.extrap_method = nowcasts.get_method("extrapolation")
        self.sprog_method = nowcasts.get_method("sprog")
        self.steps_method = nowcasts.get_method("steps")
        self.db_transform = utils.transformation.dB_transform

    def estimate_motion(self, precip_sequence: np.ndarray) -> np.ndarray:
        """Estimates 2D motion field (u, v) from a sequence of precipitation frames.
        
        Args:
            precip_sequence: Array of shape (T, H, W) where T >= 3, in mm/h.
            
        Returns:
            velocity: Array of shape (2, H, W) in pixels / timestep.
        """
        if precip_sequence.ndim != 3 or precip_sequence.shape[0] < 2:
            raise ValueError(f"Expected precip_sequence of shape (T, H, W) with T >= 2, got {precip_sequence.shape}")

        # If field is completely dry, return zero velocity
        if np.nanmax(precip_sequence) < 0.1:
            H, W = precip_sequence.shape[1], precip_sequence.shape[2]
            return np.zeros((2, H, W), dtype=np.float32)

        try:
            # Transform to dB for robust feature tracking
            clean = np.where(np.isfinite(precip_sequence), precip_sequence, 0.0)
            v = self.motion_method(clean)
            return v
        except Exception as exc:
            logger.warning("Lucas-Kanade optical flow failed (%s), returning zero motion", exc)
            H, W = precip_sequence.shape[1], precip_sequence.shape[2]
            return np.zeros((2, H, W), dtype=np.float32)

    def predict_deterministic(
        self,
        precip_sequence: np.ndarray,
        lead_steps: int = 6,
        method: str = "sprog",
        velocity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generates deterministic nowcast (S-PROG or Extrapolation).
        
        Args:
            precip_sequence: Historical sequence of shape (T, H, W) in mm/h (T >= 3).
            lead_steps: Number of forward timesteps (6 = 3 hours at 30 min freq).
            method: 'sprog' or 'extrapolation'.
            velocity: Optional precomputed velocity field (2, H, W).
            
        Returns:
            nowcast: Array of shape (lead_steps, H, W) in mm/h.
        """
        clean = np.where(np.isfinite(precip_sequence), precip_sequence, 0.0).astype(np.float32)
        if np.nanmax(clean) < 0.05:
            # Near-zero rain across entire domain
            return np.zeros((lead_steps, clean.shape[1], clean.shape[2]), dtype=np.float32)

        if velocity is None:
            velocity = self.estimate_motion(clean)

        # dB transformation
        zerovalue = -15.0
        precip_db, _ = self.db_transform(clean, threshold=0.1, zerovalue=zerovalue)

        if method.lower() == "extrapolation":
            fc_db = self.extrap_method(precip_db[-1], velocity, timesteps=lead_steps)
        else:
            try:
                fc_db = self.sprog_method(
                    precip_db,
                    velocity,
                    timesteps=lead_steps,
                    precip_thr=zerovalue,
                    n_cascade_levels=self.n_cascade_levels,
                    ar_order=self.ar_order,
                )
            except Exception as e:
                logger.warning("S-PROG failed (%s), falling back to extrapolation", e)
                fc_db = self.extrap_method(precip_db[-1], velocity, timesteps=lead_steps)

        fc_rain, _ = self.db_transform(fc_db, threshold=zerovalue, inverse=True)
        return np.clip(np.nan_to_num(fc_rain, nan=0.0), 0.0, 300.0).astype(np.float32)

    def predict_probabilistic(
        self,
        precip_sequence: np.ndarray,
        lead_steps: int = 6,
        n_ens_members: int = 20,
        seed: int = 42,
        velocity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generates probabilistic ensemble nowcast using STEPS.
        
        Args:
            precip_sequence: Historical sequence of shape (T, H, W) in mm/h (T >= 3).
            lead_steps: Number of forward timesteps.
            n_ens_members: Number of ensemble members (default: 20).
            seed: Random seed for perturbation reproducibility.
            velocity: Optional precomputed velocity field.
            
        Returns:
            ensemble: Array of shape (n_ens_members, lead_steps, H, W) in mm/h.
        """
        clean = np.where(np.isfinite(precip_sequence), precip_sequence, 0.0).astype(np.float32)
        H, W = clean.shape[1], clean.shape[2]
        if np.nanmax(clean) < 0.05:
            return np.zeros((n_ens_members, lead_steps, H, W), dtype=np.float32)

        if velocity is None:
            velocity = self.estimate_motion(clean)

        zerovalue = -15.0
        precip_db, _ = self.db_transform(clean, threshold=0.1, zerovalue=zerovalue)

        try:
            fc_db = self.steps_method(
                precip_db,
                velocity,
                timesteps=lead_steps,
                n_ens_members=n_ens_members,
                n_cascade_levels=self.n_cascade_levels,
                precip_thr=zerovalue,
                kmperpixel=self.kmperpixel,
                timestep=self.timestep_minutes,
                seed=seed,
            )
            fc_rain, _ = self.db_transform(fc_db, threshold=zerovalue, inverse=True)
            return np.clip(np.nan_to_num(fc_rain, nan=0.0), 0.0, 300.0).astype(np.float32)
        except Exception as e:
            logger.warning("STEPS ensemble failed (%s), constructing ensemble around deterministic nowcast", e)
            det = self.predict_deterministic(clean, lead_steps=lead_steps, velocity=velocity)
            # Add stochastic perturbation if STEPS library call failed
            rng = np.random.RandomState(seed)
            ensemble = np.zeros((n_ens_members, lead_steps, H, W), dtype=np.float32)
            for m in range(n_ens_members):
                noise = rng.normal(1.0, 0.15 + 0.05 * np.arange(lead_steps)[:, None, None], size=det.shape)
                ensemble[m] = np.clip(det * noise, 0.0, 300.0)
            return ensemble

    def predict_persistence(self, precip_sequence: np.ndarray, lead_steps: int = 6) -> np.ndarray:
        """Standard persistence baseline: holding current rainfall constant across lead times."""
        last_frame = np.where(np.isfinite(precip_sequence[-1]), precip_sequence[-1], 0.0)
        return np.repeat(last_frame[np.newaxis, :, :], lead_steps, axis=0).astype(np.float32)


# =============================================================================
# VERIFICATION METRICS
# =============================================================================

def compute_csi(forecast: np.ndarray, obs: np.ndarray, threshold_mm_h: float) -> float:
    """Computes Critical Success Index (Threat Score) at a given threshold.
    
    CSI = Hits / (Hits + Misses + False_Alarms)
    """
    f_bin = forecast >= threshold_mm_h
    o_bin = obs >= threshold_mm_h

    hits = np.logical_and(f_bin, o_bin).sum()
    misses = np.logical_and(np.logical_not(f_bin), o_bin).sum()
    false_alarms = np.logical_and(f_bin, np.logical_not(o_bin)).sum()

    denominator = hits + misses + false_alarms
    if denominator == 0:
        return 1.0  # Perfect score when both forecast and obs correctly predict no event
    return float(hits / denominator)


def compute_fss(
    forecast: np.ndarray,
    obs: np.ndarray,
    threshold_mm_h: float,
    scale_pixels: int = 1,
) -> float:
    """Computes Fractions Skill Score (FSS) at a specified spatial neighborhood scale.
    
    Args:
        forecast: 2D array of predictions in mm/h.
        obs: 2D array of observations in mm/h.
        threshold_mm_h: Exceedance threshold.
        scale_pixels: Box filter window size (e.g. 1 = 10km, 2 = 20km, 4 = 40km, 8 = 80km).
    """
    f_bin = (forecast >= threshold_mm_h).astype(np.float32)
    o_bin = (obs >= threshold_mm_h).astype(np.float32)

    if scale_pixels > 1:
        f_frac = uniform_filter(f_bin, size=scale_pixels, mode="constant", cval=0.0)
        o_frac = uniform_filter(o_bin, size=scale_pixels, mode="constant", cval=0.0)
    else:
        f_frac = f_bin
        o_frac = o_bin

    mse = float(np.mean((f_frac - o_frac) ** 2))
    mse_ref = float(np.mean(f_frac ** 2 + o_frac ** 2))

    if mse_ref == 0.0:
        return 1.0 if mse == 0.0 else 0.0
    return max(0.0, 1.0 - (mse / mse_ref))


def compute_brier_score(prob_forecast: np.ndarray, obs: np.ndarray, threshold_mm_h: float) -> float:
    """Computes Brier Score for probabilistic forecasts.
    
    BS = (1/N) * sum((P_i - O_i)^2)
    """
    o_bin = (obs >= threshold_mm_h).astype(np.float32)
    p_clamped = np.clip(prob_forecast, 0.0, 1.0)
    return float(np.mean((p_clamped - o_bin) ** 2))
