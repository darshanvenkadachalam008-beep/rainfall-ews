"""Unit tests for Stage 3 PySteps baseline nowcasting."""

import numpy as np
import pytest

from rews.nowcasting.baseline_pysteps import (
    PyStepsNowcaster,
    compute_brier_score,
    compute_csi,
    compute_fss,
)


@pytest.fixture
def synthetic_precip_sequence():
    """Generates a synthetic sequence of 4 precipitation frames (4, 50, 60)."""
    rng = np.random.RandomState(42)
    # Background light rain with an advecting rain band
    H, W = 50, 60
    frames = []
    for t in range(4):
        field = np.zeros((H, W), dtype=np.float32)
        # Moving rain cluster from (15 + 2*t, 20 + 3*t)
        cy, cx = 15 + 2 * t, 20 + 3 * t
        y, x = np.ogrid[:H, :W]
        dist_sq = (y - cy) ** 2 + (x - cx) ** 2
        cluster = 25.0 * np.exp(-dist_sq / (2 * 6.0 ** 2))
        noise = rng.exponential(scale=0.5, size=(H, W))
        field = (cluster + noise).astype(np.float32)
        frames.append(field)
    return np.stack(frames, axis=0)


def test_motion_estimation(synthetic_precip_sequence):
    nowcaster = PyStepsNowcaster(n_cascade_levels=3)
    velocity = nowcaster.estimate_motion(synthetic_precip_sequence)
    assert velocity.shape == (2, 50, 60)
    assert np.all(np.isfinite(velocity))
    # Motion should detect rightward/downward displacement
    assert np.any(velocity != 0.0)


def test_sprog_deterministic_nowcast(synthetic_precip_sequence):
    nowcaster = PyStepsNowcaster(n_cascade_levels=3)
    forecast = nowcaster.predict_deterministic(synthetic_precip_sequence, lead_steps=6, method="sprog")
    assert forecast.shape == (6, 50, 60)
    assert np.all(np.isfinite(forecast))
    assert np.all(forecast >= 0.0)
    # Rain should be predicted with non-trivial values
    assert float(np.max(forecast)) > 1.0


def test_steps_probabilistic_nowcast(synthetic_precip_sequence):
    nowcaster = PyStepsNowcaster(n_cascade_levels=3)
    n_ens = 5
    ensemble = nowcaster.predict_probabilistic(synthetic_precip_sequence, lead_steps=6, n_ens_members=n_ens, seed=42)
    assert ensemble.shape == (n_ens, 6, 50, 60)
    assert np.all(np.isfinite(ensemble))
    assert np.all(ensemble >= 0.0)
    # Ensemble members should have stochastic variance (not identical)
    var_across_members = np.var(ensemble, axis=0)
    assert float(np.mean(var_across_members)) > 0.0


def test_persistence_baseline(synthetic_precip_sequence):
    nowcaster = PyStepsNowcaster()
    persistence = nowcaster.predict_persistence(synthetic_precip_sequence, lead_steps=6)
    assert persistence.shape == (6, 50, 60)
    for lead in range(6):
        assert np.array_equal(persistence[lead], synthetic_precip_sequence[-1])


def test_verification_csi_metric():
    obs = np.array([[10.0, 0.0], [5.0, 0.0]])
    fc_good = np.array([[12.0, 0.0], [6.0, 0.0]])
    fc_bad = np.array([[0.0, 10.0], [0.0, 5.0]])

    csi_good = compute_csi(fc_good, obs, threshold_mm_h=2.5)
    csi_bad = compute_csi(fc_bad, obs, threshold_mm_h=2.5)

    assert csi_good == 1.0
    assert csi_bad == 0.0


def test_verification_fss_metric():
    obs = np.zeros((30, 30), dtype=np.float32)
    obs[10:15, 10:15] = 20.0  # 5x5 rain patch

    # Perfect forecast
    fss_perfect = compute_fss(obs, obs, threshold_mm_h=5.0, scale_pixels=1)
    assert fss_perfect == 1.0

    # Shifted forecast by 2 pixels
    shifted = np.zeros((30, 30), dtype=np.float32)
    shifted[12:17, 12:17] = 20.0
    fss_scale1 = compute_fss(shifted, obs, threshold_mm_h=5.0, scale_pixels=1)
    fss_scale4 = compute_fss(shifted, obs, threshold_mm_h=5.0, scale_pixels=4)

    # Neighborhood smoothing should increase skill for shifted rain
    assert fss_scale4 > fss_scale1


def test_verification_brier_score():
    obs = np.array([[15.0, 0.0], [0.0, 20.0]])
    perfect_prob = np.array([[1.0, 0.0], [0.0, 1.0]])
    worst_prob = np.array([[0.0, 1.0], [1.0, 0.0]])

    bs_perfect = compute_brier_score(perfect_prob, obs, threshold_mm_h=10.0)
    bs_worst = compute_brier_score(worst_prob, obs, threshold_mm_h=10.0)

    assert bs_perfect == 0.0
    assert bs_worst == 1.0
