"""
Unit tests for signal generation utilities.

This module tests:
- tone signal generation
- burst signal generation
- steering vector computation
- noise addition
- multi-channel array data generation
"""


import numpy as np

from src.signal_generator import (
    generate_tone_signal,
    generate_burst_signal,
    steering_vector_ula,
    add_noise,
    generate_array_data,
    generate_array_data_multi,
)


def test_generate_tone_signal_shape():
    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    assert signal.shape == (128,)


def test_generate_tone_signal_is_complex():
    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    assert np.iscomplexobj(signal)


def test_generate_tone_signal_unit_magnitude():
    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    assert np.allclose(np.abs(signal), 1.0)

def test_generate_burst_signal_shape():
    signal = generate_burst_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
        burst_start=20,
        burst_length=40,
    )
    assert signal.shape == (128,)


def test_generate_burst_signal_zero_outside_burst():
    signal = generate_burst_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
        burst_start=20,
        burst_length=40,
    )
    assert np.allclose(signal[:20], 0.0)
    assert np.allclose(signal[60:], 0.0)


def test_generate_burst_signal_nonzero_inside_burst():
    signal = generate_burst_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
        burst_start=20,
        burst_length=40,
    )
    assert np.all(np.abs(signal[20:60]) > 0.0)


def test_steering_vector_shape():
    sv = steering_vector_ula(angle_deg=0.0)
    assert sv.shape == (4,)


def test_steering_vector_zero_angle():
    sv = steering_vector_ula(angle_deg=0.0)
    # All elements should be 1 (no phase shift)
    assert np.allclose(sv, 1.0)


def test_steering_vector_nonzero_angle_phase_progression():
    sv = steering_vector_ula(angle_deg=30.0)
    # Adjacent elements should not be equal (phase shift exists)
    assert not np.allclose(sv[0], sv[1])


def test_add_noise_shape():
    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    noisy_signal = add_noise(signal, snr_db=10.0)
    assert noisy_signal.shape == signal.shape


def test_add_noise_changes_signal():
    rng = np.random.default_rng(42)
    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    noisy_signal = add_noise(signal, snr_db=10.0, rng=rng)
    assert not np.allclose(noisy_signal, signal)


def test_add_noise_zero_signal_returns_noise():
    rng = np.random.default_rng(42)
    signal = np.zeros(128, dtype=np.complex128)
    noisy_signal = add_noise(signal, snr_db=10.0, rng=rng)
    assert noisy_signal.shape == signal.shape
    assert np.iscomplexobj(noisy_signal)


def test_generate_array_data_shape():
    rng = np.random.default_rng(42)
    source_signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    array_data = generate_array_data(
        source_signal=source_signal,
        angle_deg=20.0,
        snr_db=10.0,
        rng=rng,
    )
    assert array_data.shape == (4, 128)


def test_generate_array_data_is_complex():
    rng = np.random.default_rng(42)
    source_signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    array_data = generate_array_data(
        source_signal=source_signal,
        angle_deg=20.0,
        snr_db=10.0,
        rng=rng,
    )
    assert np.iscomplexobj(array_data)


def test_generate_array_data_channels_differ_for_nonzero_angle():
    rng = np.random.default_rng(42)
    source_signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    array_data = generate_array_data(
        source_signal=source_signal,
        angle_deg=30.0,
        snr_db=30.0,
        rng=rng,
    )
    assert not np.allclose(array_data[0], array_data[1])


def test_generate_array_data_multi_shape():
    rng = np.random.default_rng(42)

    signal_1 = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    signal_2 = generate_tone_signal(
        frequency_hz=1500.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )

    emitters = [
        {"source_signal": signal_1, "angle_deg": -20.0, "amplitude": 1.0},
        {"source_signal": signal_2, "angle_deg": 35.0, "amplitude": 0.8},
    ]

    array_data = generate_array_data_multi(
        emitters=emitters,
        snr_db=20.0,
        rng=rng,
    )

    assert array_data.shape == (4, 128)


def test_generate_array_data_multi_is_complex():
    rng = np.random.default_rng(42)

    signal_1 = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    signal_2 = generate_tone_signal(
        frequency_hz=1200.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )

    emitters = [
        {"source_signal": signal_1, "angle_deg": -10.0, "amplitude": 1.0},
        {"source_signal": signal_2, "angle_deg": 25.0, "amplitude": 0.5},
    ]

    array_data = generate_array_data_multi(
        emitters=emitters,
        snr_db=15.0,
        rng=rng,
    )

    assert np.iscomplexobj(array_data)


def test_generate_array_data_multi_channels_differ():
    rng = np.random.default_rng(42)

    signal_1 = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    signal_2 = generate_tone_signal(
        frequency_hz=1600.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )

    emitters = [
        {"source_signal": signal_1, "angle_deg": -30.0, "amplitude": 1.0},
        {"source_signal": signal_2, "angle_deg": 30.0, "amplitude": 1.0},
    ]

    array_data = generate_array_data_multi(
        emitters=emitters,
        snr_db=25.0,
        rng=rng,
    )

    assert not np.allclose(array_data[0], array_data[1])


def test_generate_array_data_multi_differs_from_single_emitter():
    rng_single = np.random.default_rng(42)
    rng_multi = np.random.default_rng(42)

    signal_1 = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    signal_2 = generate_tone_signal(
        frequency_hz=1500.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )

    single_array_data = generate_array_data(
        source_signal=signal_1,
        angle_deg=-20.0,
        snr_db=20.0,
        rng=rng_single,
    )

    emitters = [
        {"source_signal": signal_1, "angle_deg": -20.0, "amplitude": 1.0},
        {"source_signal": signal_2, "angle_deg": 35.0, "amplitude": 0.8},
    ]

    multi_array_data = generate_array_data_multi(
        emitters=emitters,
        snr_db=20.0,
        rng=rng_multi,
    )

    assert not np.allclose(single_array_data, multi_array_data)


def test_generate_array_data_multi_empty_emitters_raises():
    try:
        generate_array_data_multi(
            emitters=[],
            snr_db=10.0,
        )
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_generate_array_data_multi_mismatched_lengths_raises():
    signal_1 = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )
    signal_2 = generate_tone_signal(
        frequency_hz=1500.0,
        sample_rate_hz=10000.0,
        num_samples=64,
    )

    emitters = [
        {"source_signal": signal_1, "angle_deg": -20.0, "amplitude": 1.0},
        {"source_signal": signal_2, "angle_deg": 35.0, "amplitude": 0.8},
    ]

    try:
        generate_array_data_multi(
            emitters=emitters,
            snr_db=10.0,
        )
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_generate_array_data_multi_negative_amplitude_raises():
    signal = generate_tone_signal(
        frequency_hz=1000.0,
        sample_rate_hz=10000.0,
        num_samples=128,
    )

    emitters = [
        {"source_signal": signal, "angle_deg": 10.0, "amplitude": -1.0},
    ]

    try:
        generate_array_data_multi(
            emitters=emitters,
            snr_db=10.0,
        )
        assert False, "Expected ValueError"
    except ValueError:
        assert True