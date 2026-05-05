"""
Unit tests for preprocessing and detection modules.

Covers:
- DC removal
- signal normalization
- energy computation
- threshold-based signal detection
"""

import numpy as np

from src.preprocessing import remove_dc, normalize_signal
from src.detection import (
    compute_signal_energy,
    detect_signal_energy,
    ca_cfar_1d,
    find_cfar_confirmed_peaks,
)
from src.signal_generator import generate_tone_signal, generate_array_data
from src.metrics import (
    compute_peak_power,
    compute_peak_to_average_ratio,
    compute_peak_to_sidelobe_ratio,
    compute_confidence_score,
)


def test_remove_dc_1d():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = remove_dc(x)
    assert np.isclose(np.mean(y), 0.0)


def test_remove_dc_2d():
    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ],
        dtype=np.float64,
    )
    y = remove_dc(x)
    assert np.allclose(np.mean(y, axis=1), 0.0)


def test_normalize_signal_1d():
    x = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    y = normalize_signal(x)
    assert np.isclose(np.max(np.abs(y)), 1.0)


def test_normalize_signal_2d():
    x = np.array(
        [
            [1.0, -2.0, 0.5],
            [4.0, -1.0, 2.0],
        ],
        dtype=np.float64,
    )
    y = normalize_signal(x)
    assert np.isclose(np.max(np.abs(y)), 1.0)


def test_normalize_signal_zero_input():
    x = np.zeros(5, dtype=np.float64)
    y = normalize_signal(x)
    assert np.allclose(y, 0.0)


def test_compute_signal_energy_1d():
    x = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    energy = compute_signal_energy(x)
    assert np.isclose(energy, 1.0)


def test_compute_signal_energy_complex():
    x = np.array([1.0 + 1.0j, 1.0 + 1.0j], dtype=np.complex128)
    energy = compute_signal_energy(x)
    assert np.isclose(energy, 2.0)


def test_detect_signal_energy_true():
    x = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    detected, energy = detect_signal_energy(x, threshold=0.5)
    assert detected is True
    assert np.isclose(energy, 1.0)


def test_detect_signal_energy_false():
    x = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    detected, energy = detect_signal_energy(x, threshold=0.5)
    assert detected is False
    assert np.isclose(energy, 0.01)


def test_detection_noise_only():
    rng = np.random.default_rng(42)

    # pure noise (zero signal input → add_noise generates noise)
    noise = rng.standard_normal(128) + 1j * rng.standard_normal(128)

    noise = remove_dc(noise)
    noise = normalize_signal(noise)

    detected, energy = detect_signal_energy(noise, threshold=0.2)

    assert detected is False


def test_detection_signal_present():
    rng = np.random.default_rng(42)

    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=20.0,
        snr_db=10.0,
        rng=rng,
    )

    # use first channel
    x = array_data[0]

    x = remove_dc(x)
    x = normalize_signal(x)

    detected, energy = detect_signal_energy(x, threshold=0.2)

    assert detected is True


def test_compute_peak_power():
    spectrum = np.array([0.1, 0.5, 0.2])
    peak = compute_peak_power(spectrum)
    assert peak == 0.5


def test_peak_to_average_ratio():
    spectrum = np.array([1.0, 1.0, 1.0])
    par = compute_peak_to_average_ratio(spectrum)
    assert np.isclose(par, 1.0)


def test_peak_to_sidelobe_ratio():
    spectrum = np.array([0.1, 0.8, 0.5])
    pslr = compute_peak_to_sidelobe_ratio(spectrum)
    assert np.isclose(pslr, 0.8 / 0.5)

def test_compute_confidence_score():
    spectrum = np.array([0.1, 1.0, 0.2])
    score = compute_confidence_score(spectrum)

    assert score > 1.0

def test_ca_cfar_1d_output_shapes():
    x = np.array([0.1, 0.2, 1.0, 0.2, 0.1, 0.3, 0.2], dtype=np.float64)

    detections, thresholds = ca_cfar_1d(
        x=x,
        num_training_cells=1,
        num_guard_cells=0,
        threshold_scale=2.0,
    )

    assert detections.shape == x.shape
    assert thresholds.shape == x.shape


def test_ca_cfar_1d_detects_strong_peak():
    x = np.array([0.1, 0.2, 1.5, 0.2, 0.1], dtype=np.float64)

    detections, thresholds = ca_cfar_1d(
        x=x,
        num_training_cells=1,
        num_guard_cells=0,
        threshold_scale=3.0,
    )

    assert detections[2] is np.True_ or detections[2] == True


def test_ca_cfar_1d_no_detection_for_flat_spectrum():
    x = np.ones(9, dtype=np.float64)

    detections, thresholds = ca_cfar_1d(
        x=x,
        num_training_cells=2,
        num_guard_cells=1,
        threshold_scale=1.5,
    )

    assert not np.any(detections[np.isfinite(thresholds)])


def test_find_cfar_confirmed_peaks_returns_local_max_only():
    x = np.array([0.1, 0.3, 1.2, 0.4, 0.2, 0.9, 0.1], dtype=np.float64)
    detections = np.array([False, False, True, False, False, True, False])

    peak_indices = find_cfar_confirmed_peaks(x, detections)

    assert np.array_equal(peak_indices, np.array([2, 5]))


def test_find_cfar_confirmed_peaks_rejects_nonmax_detection():
    x = np.array([0.1, 0.8, 0.9, 0.2, 0.1], dtype=np.float64)
    detections = np.array([False, True, False, False, False])

    peak_indices = find_cfar_confirmed_peaks(x, detections)

    assert len(peak_indices) == 0


def test_ca_cfar_on_synthetic_spatial_spectrum_like_input():
    x = np.array(
        [0.05, 0.07, 0.06, 0.08, 1.20, 0.09, 0.07, 0.06, 0.90, 0.05, 0.04],
        dtype=np.float64,
    )

    detections, _ = ca_cfar_1d(
        x=x,
        num_training_cells=2,
        num_guard_cells=1,
        threshold_scale=3.0,
    )

    peak_indices = find_cfar_confirmed_peaks(x, detections)

    assert 4 in peak_indices or 8 in peak_indices