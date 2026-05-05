"""
Unit tests for beamforming and angle-of-arrival (AoA) estimation.

This module tests:
- spatial spectrum computation using conventional beamforming
- spectrum properties (shape, real-valued, non-negative)
- peak-based bearing estimation from spatial spectrum
- end-to-end AoA validation using synthetic array data
"""

import numpy as np

from src.beamforming import (
    beamform_response,
    compute_sample_covariance,
    estimate_bearing_from_phase_difference,
    estimate_bearing_from_spectrum,
    estimate_bearings_from_spectrum,
    find_spectrum_peaks,
    get_noise_subspace,
    music_spectrum,
)
from src.signal_generator import generate_array_data, generate_tone_signal


def test_beamform_response_shape():
    rng = np.random.default_rng(42)

    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=20.0,
        snr_db=20.0,
        rng=rng,
    )

    angle_grid = np.arange(-90, 91, 1)
    spectrum = beamform_response(array_data, angle_grid_deg=angle_grid)

    assert spectrum.shape == (len(angle_grid),)


def test_beamform_response_is_real():
    rng = np.random.default_rng(42)

    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=20.0,
        snr_db=20.0,
        rng=rng,
    )

    spectrum = beamform_response(array_data)
    assert np.isrealobj(spectrum)


def test_beamform_response_nonnegative():
    rng = np.random.default_rng(42)

    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=20.0,
        snr_db=20.0,
        rng=rng,
    )

    spectrum = beamform_response(array_data)
    assert np.all(spectrum >= 0.0)


def test_estimate_bearing_from_spectrum_returns_peak_angle():
    angle_grid = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0])
    spatial_spectrum = np.array([0.1, 0.2, 0.4, 0.9, 0.3, 0.1])

    estimated_angle, peak_value = estimate_bearing_from_spectrum(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spatial_spectrum,
    )

    assert estimated_angle == 0.0
    assert peak_value == 0.9


def test_estimate_bearing_from_spectrum_length_mismatch():
    angle_grid = np.array([-10.0, 0.0, 10.0])
    spatial_spectrum = np.array([0.2, 0.8])

    try:
        estimate_bearing_from_spectrum(
            angle_grid_deg=angle_grid,
            spatial_spectrum=spatial_spectrum,
        )
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_end_to_end_bearing_estimation_zero_deg():
    rng = np.random.default_rng(42)

    true_angle_deg = 0.0
    angle_grid = np.arange(-90, 91, 1)

    source_signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=256,
    )

    array_data = generate_array_data(
        source_signal=source_signal,
        angle_deg=true_angle_deg,
        snr_db=20.0,
        rng=rng,
    )

    spatial_spectrum = beamform_response(
        array_data=array_data,
        angle_grid_deg=angle_grid,
    )

    estimated_angle_deg, peak_value = estimate_bearing_from_spectrum(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spatial_spectrum,
    )

    assert abs(estimated_angle_deg - true_angle_deg) <= 1.0
    assert peak_value >= 0.0


def test_end_to_end_bearing_estimation_positive_angle():
    rng = np.random.default_rng(42)

    true_angle_deg = 25.0
    angle_grid = np.arange(-90, 91, 1)

    source_signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=256,
    )

    array_data = generate_array_data(
        source_signal=source_signal,
        angle_deg=true_angle_deg,
        snr_db=20.0,
        rng=rng,
    )

    spatial_spectrum = beamform_response(
        array_data=array_data,
        angle_grid_deg=angle_grid,
    )

    estimated_angle_deg, peak_value = estimate_bearing_from_spectrum(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spatial_spectrum,
    )

    assert abs(estimated_angle_deg - true_angle_deg) <= 2.0
    assert peak_value >= 0.0


def test_end_to_end_bearing_estimation_negative_angle():
    rng = np.random.default_rng(42)

    true_angle_deg = -40.0
    angle_grid = np.arange(-90, 91, 1)

    source_signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=256,
    )

    array_data = generate_array_data(
        source_signal=source_signal,
        angle_deg=true_angle_deg,
        snr_db=20.0,
        rng=rng,
    )

    spatial_spectrum = beamform_response(
        array_data=array_data,
        angle_grid_deg=angle_grid,
    )

    estimated_angle_deg, peak_value = estimate_bearing_from_spectrum(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spatial_spectrum,
    )

    assert abs(estimated_angle_deg - true_angle_deg) <= 2.0
    assert peak_value >= 0.0

def test_track_bearing_over_time_shape():
    rng = np.random.default_rng(42)

    true_angle_deg = 20.0
    angle_grid = np.arange(-90, 91, 1)

    source_signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=256,
    )

    array_data = generate_array_data(
        source_signal=source_signal,
        angle_deg=true_angle_deg,
        snr_db=20.0,
        rng=rng,
    )

    frame_indices, estimated_angles_deg = track_bearing_over_time(
        array_data=array_data,
        angle_grid_deg=angle_grid,
        frame_length=64,
    )

    assert frame_indices.shape == (4,)
    assert estimated_angles_deg.shape == (4,)


def test_track_bearing_over_time_near_true_angle():
    rng = np.random.default_rng(42)

    true_angle_deg = 20.0
    angle_grid = np.arange(-90, 91, 1)

    source_signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=256,
    )

    array_data = generate_array_data(
        source_signal=source_signal,
        angle_deg=true_angle_deg,
        snr_db=20.0,
        rng=rng,
    )

    _, estimated_angles_deg = track_bearing_over_time(
        array_data=array_data,
        angle_grid_deg=angle_grid,
        frame_length=64,
    )

    assert np.all(np.abs(estimated_angles_deg - true_angle_deg) <= 2.0)

def test_find_spectrum_peaks_returns_expected_indices():
    angle_grid = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
    spectrum = np.array([0.1, 0.9, 0.2, 0.1, 0.8, 0.3, 0.1])

    peak_indices = find_spectrum_peaks(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spectrum,
        min_separation_deg=5.0,
        max_peaks=None,
        min_peak_height=0.5,
    )

    assert np.array_equal(peak_indices, np.array([1, 4]))


def test_find_spectrum_peaks_respects_max_peaks():
    angle_grid = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
    spectrum = np.array([0.1, 0.9, 0.2, 0.1, 0.8, 0.3, 0.1])

    peak_indices = find_spectrum_peaks(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spectrum,
        min_separation_deg=5.0,
        max_peaks=1,
        min_peak_height=0.5,
    )

    assert np.array_equal(peak_indices, np.array([1]))


def test_find_spectrum_peaks_respects_min_separation():
    angle_grid = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    spectrum = np.array([0.1, 0.8, 0.2, 0.75, 0.2, 0.1, 0.05])

    peak_indices = find_spectrum_peaks(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spectrum,
        min_separation_deg=3.0,
        max_peaks=None,
        min_peak_height=0.5,
    )

    assert len(peak_indices) == 1
    assert peak_indices[0] == 1


def test_estimate_bearings_from_spectrum_returns_multiple_angles():
    angle_grid = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
    spectrum = np.array([0.1, 0.9, 0.2, 0.1, 0.8, 0.3, 0.1])

    estimated_angles, peak_values = estimate_bearings_from_spectrum(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spectrum,
        min_separation_deg=5.0,
        max_peaks=None,
        min_peak_height=0.5,
    )

    assert np.array_equal(estimated_angles, np.array([-20.0, 10.0]))
    assert np.allclose(peak_values, np.array([0.9, 0.8]))


def test_beamforming_multi_emitter_returns_at_least_two_candidates_when_separated():
    rng = np.random.default_rng(42)

    signal_1 = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=512,
    )
    signal_2 = generate_tone_signal(
        frequency_hz=1600.0,
        sample_rate_hz=10000.0,
        num_samples=512,
    )

    from src.signal_generator import generate_array_data_multi

    emitters = [
        {"source_signal": signal_1, "angle_deg": -40.0, "amplitude": 1.0},
        {"source_signal": signal_2, "angle_deg": 35.0, "amplitude": 1.0},
    ]

    array_data = generate_array_data_multi(
        emitters=emitters,
        snr_db=25.0,
        rng=rng,
    )

    angle_grid = np.arange(-90, 91, 1)
    spectrum = beamform_response(array_data, angle_grid_deg=angle_grid)

    estimated_angles, peak_values = estimate_bearings_from_spectrum(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spectrum,
        min_separation_deg=10.0,
        max_peaks=4,
        min_peak_height=0.5 * np.max(spectrum),
    )

    assert len(estimated_angles) >= 2
    assert len(peak_values) == len(estimated_angles)

def test_compute_sample_covariance_shape():
    rng = np.random.default_rng(42)

    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=256,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=20.0,
        snr_db=20.0,
        rng=rng,
    )

    covariance = compute_sample_covariance(array_data)
    assert covariance.shape == (4, 4)


def test_compute_sample_covariance_is_hermitian():
    rng = np.random.default_rng(42)

    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=256,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=10.0,
        snr_db=20.0,
        rng=rng,
    )

    covariance = compute_sample_covariance(array_data)
    assert np.allclose(covariance, np.conjugate(covariance.T))


def test_get_noise_subspace_shapes():
    rng = np.random.default_rng(42)

    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=256,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=15.0,
        snr_db=20.0,
        rng=rng,
    )

    covariance = compute_sample_covariance(array_data)
    eigenvalues, noise_subspace = get_noise_subspace(covariance, num_sources=1)

    assert eigenvalues.shape == (4,)
    assert noise_subspace.shape == (4, 3)


def test_music_spectrum_shape():
    rng = np.random.default_rng(42)

    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=256,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=25.0,
        snr_db=20.0,
        rng=rng,
    )

    angle_grid = np.arange(-90, 91, 1)
    spectrum = music_spectrum(
        array_data=array_data,
        num_sources=1,
        angle_grid_deg=angle_grid,
    )

    assert spectrum.shape == (len(angle_grid),)
    assert np.all(spectrum > 0.0)


def test_music_single_source_estimation_near_true_angle():
    rng = np.random.default_rng(42)

    true_angle_deg = 25.0
    angle_grid = np.arange(-90, 91, 1)

    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=512,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=true_angle_deg,
        snr_db=20.0,
        rng=rng,
    )

    spectrum = music_spectrum(
        array_data=array_data,
        num_sources=1,
        angle_grid_deg=angle_grid,
    )

    estimated_angle_deg, _ = estimate_bearing_from_spectrum(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spectrum,
    )

    assert abs(estimated_angle_deg - true_angle_deg) <= 2.0


def test_music_two_source_estimation_returns_two_candidates_when_separated():
    rng = np.random.default_rng(42)

    signal_1 = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=1024,
    )
    signal_2 = generate_tone_signal(
        frequency_hz=1700.0,
        sample_rate_hz=10000.0,
        num_samples=1024,
    )

    from src.signal_generator import generate_array_data_multi

    emitters = [
        {"source_signal": signal_1, "angle_deg": -35.0, "amplitude": 1.0},
        {"source_signal": signal_2, "angle_deg": 30.0, "amplitude": 1.0},
    ]

    array_data = generate_array_data_multi(
        emitters=emitters,
        snr_db=25.0,
        rng=rng,
    )

    angle_grid = np.arange(-90, 91, 1)
    spectrum = music_spectrum(
        array_data=array_data,
        num_sources=2,
        angle_grid_deg=angle_grid,
    )

    estimated_angles, peak_values = estimate_bearings_from_spectrum(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spectrum,
        min_separation_deg=8.0,
        max_peaks=2,
        min_peak_height=None,
    )

    print("Estimated angles:", estimated_angles)
    print("Peak values:", peak_values)

    assert len(estimated_angles) >= 2
    assert len(peak_values) == len(estimated_angles)


def test_music_two_source_contains_angles_near_truth():
    rng = np.random.default_rng(42)

    true_angles = np.array([-30.0, 35.0])

    signal_1 = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=1024,
    )
    signal_2 = generate_tone_signal(
        frequency_hz=1800.0,
        sample_rate_hz=10000.0,
        num_samples=1024,
    )

    from src.signal_generator import generate_array_data_multi

    emitters = [
        {"source_signal": signal_1, "angle_deg": true_angles[0], "amplitude": 1.0},
        {"source_signal": signal_2, "angle_deg": true_angles[1], "amplitude": 1.0},
    ]

    array_data = generate_array_data_multi(
        emitters=emitters,
        snr_db=25.0,
        rng=rng,
    )

    angle_grid = np.arange(-90, 91, 1)
    spectrum = music_spectrum(
        array_data=array_data,
        num_sources=2,
        angle_grid_deg=angle_grid,
    )

    estimated_angles, _ = estimate_bearings_from_spectrum(
        angle_grid_deg=angle_grid,
        spatial_spectrum=spectrum,
        min_separation_deg=10.0,
        max_peaks=4,
        min_peak_height= 0.15 * np.max(spectrum),
    )

    assert np.any(np.abs(estimated_angles - true_angles[0]) <= 3.0)
    assert np.any(np.abs(estimated_angles - true_angles[1]) <= 3.0)

def test_phase_difference_zero_deg():
    rng = np.random.default_rng(42)

    true_angle_deg = 0.0
    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=512,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=true_angle_deg,
        snr_db=25.0,
        rng=rng,
    )

    estimated_angle_deg, mean_phase_diff_rad = estimate_bearing_from_phase_difference(array_data)

    assert abs(estimated_angle_deg - true_angle_deg) <= 1.0
    assert abs(mean_phase_diff_rad) <= 0.2


def test_phase_difference_positive_angle():
    rng = np.random.default_rng(42)

    true_angle_deg = 20.0
    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=512,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=true_angle_deg,
        snr_db=25.0,
        rng=rng,
    )

    estimated_angle_deg, _ = estimate_bearing_from_phase_difference(array_data)

    assert abs(estimated_angle_deg - true_angle_deg) <= 2.0


def test_phase_difference_negative_angle():
    rng = np.random.default_rng(42)

    true_angle_deg = -30.0
    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=512,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=true_angle_deg,
        snr_db=25.0,
        rng=rng,
    )

    estimated_angle_deg, _ = estimate_bearing_from_phase_difference(array_data)

    assert abs(estimated_angle_deg - true_angle_deg) <= 2.0


def test_phase_difference_noisy_single_source():
    rng = np.random.default_rng(42)

    true_angle_deg = 25.0
    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=1024,
    )

    array_data = generate_array_data(
        source_signal=signal,
        angle_deg=true_angle_deg,
        snr_db=5.0,
        rng=rng,
    )

    estimated_angle_deg, _ = estimate_bearing_from_phase_difference(array_data)

    assert abs(estimated_angle_deg - true_angle_deg) <= 5.0