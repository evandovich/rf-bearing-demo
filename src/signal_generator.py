"""

RFDetect Lab - Signal Processing Module

Author: Evans Baidoo
Copyright (c) 2026 Evans Baidoo

Signal generation utilities for synthetic multi-channel IQ data.

"""

import numpy as np
from config import NUM_ANTENNAS, ANTENNA_SPACING_M, WAVELENGTH_M


def generate_tone_signal(
    frequency_hz: float,
    sample_rate_hz: float,
    num_samples: int,
) -> np.ndarray:
    """
    Generate a complex baseband tone.

    Args:
        frequency_hz: Tone frequency in Hz.
        sample_rate_hz: Sampling rate in Hz.
        num_samples: Number of samples to generate.

    Returns:
        Complex numpy array of shape (num_samples,).
    """
    n = np.arange(num_samples)
    signal = np.exp(1j * 2.0 * np.pi * frequency_hz * n / sample_rate_hz)
    return signal


def generate_burst_signal(
    frequency_hz: float,
    sample_rate_hz: float,
    num_samples: int,
    burst_start: int,
    burst_length: int,
) -> np.ndarray:
    """
    Generate a complex baseband burst signal.

    A tone is generated and gated so that it is only active over
    a selected range of samples.

    Args:
        frequency_hz: Tone frequency in Hz.
        sample_rate_hz: Sampling rate in Hz.
        num_samples: Total number of samples in the frame.
        burst_start: Start index of the burst.
        burst_length: Number of active burst samples.

    Returns:
        Complex numpy array of shape (num_samples,).
    """
    signal = np.zeros(num_samples, dtype=np.complex128)

    burst_end = min(burst_start + burst_length, num_samples)
    if burst_start >= num_samples or burst_length <= 0:
        return signal

    n = np.arange(burst_end - burst_start)
    burst = np.exp(1j * 2.0 * np.pi * frequency_hz * n / sample_rate_hz)

    signal[burst_start:burst_end] = burst
    return signal


def steering_vector_ula(
    angle_deg: float,
    num_antennas: int = NUM_ANTENNAS,
    antenna_spacing_m: float = ANTENNA_SPACING_M,
    wavelength_m: float = WAVELENGTH_M,
) -> np.ndarray:
    """
    Compute the steering vector for a ULA.

    Args:
        angle_deg: Angle of arrival in degrees.
        num_antennas: Number of antennas.
        antenna_spacing_m: Distance between antennas.
        wavelength_m: Signal wavelength.

    Returns:
        Steering vector of shape (num_antennas,)
    """
    theta_rad = np.deg2rad(angle_deg)

    m = np.arange(num_antennas)

    phase_shift = -2.0 * np.pi * m * antenna_spacing_m * np.sin(theta_rad) / wavelength_m

    steering = np.exp(1j * phase_shift)

    return steering


def add_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add complex Gaussian noise to a signal for a desired SNR.

    Args:
        signal: Input complex signal array.
        snr_db: Desired signal-to-noise ratio in dB.
        rng: Optional NumPy random generator.

    Returns:
        Noisy complex signal array with the same shape as input.
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(np.abs(signal) ** 2)

    if signal_power == 0.0:
        noise_power: float = 0.01,
        noise =  np.sqrt(noise_power) *(
            rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
        ) / np.sqrt(2.0)
        return noise

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    ) / np.sqrt(2.0)

    return signal + noise


def _validate_emitters(emitters: list[dict]) -> int:
    """
    Validate a list of emitter dictionaries.

    Each emitter must contain:
    - source_signal: 1D complex numpy array
    - angle_deg: finite float/int
    - amplitude: nonnegative float/int

    Returns:
        Number of samples shared by all emitters.
    """
    if not emitters:
        raise ValueError("emitters must be a non-empty list.")

    required_keys = {"source_signal", "angle_deg", "amplitude"}

    num_samples = None

    for idx, emitter in enumerate(emitters):
        if not isinstance(emitter, dict):
            raise ValueError(f"Emitter at index {idx} must be a dictionary.")

        missing_keys = required_keys - set(emitter.keys())
        if missing_keys:
            raise ValueError(
                f"Emitter at index {idx} is missing required keys: {sorted(missing_keys)}"
            )

        source_signal = emitter["source_signal"]
        angle_deg = emitter["angle_deg"]
        amplitude = emitter["amplitude"]

        if not isinstance(source_signal, np.ndarray):
            raise ValueError(f"Emitter at index {idx} has invalid source_signal type.")

        if source_signal.ndim != 1:
            raise ValueError(f"Emitter at index {idx} source_signal must be 1D.")

        if num_samples is None:
            num_samples = source_signal.shape[0]
        elif source_signal.shape[0] != num_samples:
            raise ValueError("All emitter source_signal arrays must have the same length.")

        if not np.isfinite(angle_deg):
            raise ValueError(f"Emitter at index {idx} has non-finite angle_deg.")

        if not np.isfinite(amplitude):
            raise ValueError(f"Emitter at index {idx} has non-finite amplitude.")

        if amplitude < 0.0:
            raise ValueError(f"Emitter at index {idx} amplitude must be nonnegative.")

    return int(num_samples)


def generate_array_data(
    source_signal: np.ndarray,
    angle_deg: float,
    snr_db: float,
    num_antennas: int = NUM_ANTENNAS,
    antenna_spacing_m: float = ANTENNA_SPACING_M,
    wavelength_m: float = WAVELENGTH_M,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate synthetic multi-channel array data for a single emitter.

    Args:
        source_signal: Complex source signal of shape (num_samples,).
        angle_deg: Angle of arrival in degrees.
        snr_db: Desired SNR in dB.
        num_antennas: Number of antennas.
        antenna_spacing_m: Antenna spacing in meters.
        wavelength_m: Wavelength in meters.
        rng: Optional NumPy random generator.

    Returns:
        Array data of shape (num_antennas, num_samples).
    """
    steering = steering_vector_ula(
        angle_deg=angle_deg,
        num_antennas=num_antennas,
        antenna_spacing_m=antenna_spacing_m,
        wavelength_m=wavelength_m,
    )

    array_data = steering[:, np.newaxis] * source_signal[np.newaxis, :]

    noisy_array_data = np.empty_like(array_data, dtype=np.complex128)
    for m in range(num_antennas):
        noisy_array_data[m] = add_noise(array_data[m], snr_db=snr_db, rng=rng)

    return noisy_array_data


def generate_array_data_multi(
    emitters: list[dict],
    snr_db: float,
    num_antennas: int = NUM_ANTENNAS,
    antenna_spacing_m: float = ANTENNA_SPACING_M,
    wavelength_m: float = WAVELENGTH_M,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate synthetic multi-channel array data for multiple emitters.

    Each emitter dictionary must contain:
    - source_signal: complex array of shape (num_samples,)
    - angle_deg: angle of arrival in degrees
    - amplitude: linear amplitude scale

    Args:
        emitters: List of emitter dictionaries.
        snr_db: Desired SNR in dB, applied after summing all emitters.
        num_antennas: Number of antennas.
        antenna_spacing_m: Antenna spacing in meters.
        wavelength_m: Wavelength in meters.
        rng: Optional NumPy random generator.

    Returns:
        Array data of shape (num_antennas, num_samples).
    """
    num_samples = _validate_emitters(emitters)

    if rng is None:
        rng = np.random.default_rng()

    combined_array_data = np.zeros((num_antennas, num_samples), dtype=np.complex128)

    for emitter in emitters:
        source_signal = emitter["source_signal"]
        angle_deg = float(emitter["angle_deg"])
        amplitude = float(emitter["amplitude"])

        steering = steering_vector_ula(
            angle_deg=angle_deg,
            num_antennas=num_antennas,
            antenna_spacing_m=antenna_spacing_m,
            wavelength_m=wavelength_m,
        )

        combined_array_data += (
            amplitude * steering[:, np.newaxis] * source_signal[np.newaxis, :]
        )

    noisy_array_data = np.empty_like(combined_array_data, dtype=np.complex128)
    for m in range(num_antennas):
        noisy_array_data[m] = add_noise(combined_array_data[m], snr_db=snr_db, rng=rng)

    return noisy_array_data