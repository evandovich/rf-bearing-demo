"""

RFDetect Lab - Signal Processing Module

Author: Evans Baidoo
Copyright (c) 2026 Evans Baidoo


Spectrum and spectrogram analysis utilities.
"""

import numpy as np
from scipy.signal import get_window, spectrogram


SUPPORTED_WINDOW_TYPES = [
    "rectangular",
    "hann",
    "hamming",
    "blackman",
]


def _resolve_window(window_type: str, num_points: int) -> np.ndarray:
    """
    Create a window array from a user-facing window type.

    Args:
        window_type: Name of the window.
        num_points: Window length.

    Returns:
        1D NumPy array containing the selected window.
    """
    if num_points < 1:
        raise ValueError("num_points must be positive.")

    normalized_window_type = window_type.lower()

    if normalized_window_type == "rectangular":
        return np.ones(num_points, dtype=np.float64)

    if normalized_window_type not in SUPPORTED_WINDOW_TYPES:
        raise ValueError(
            f"Unsupported window_type '{window_type}'. "
            f"Supported options: {SUPPORTED_WINDOW_TYPES}"
        )

    window_array = get_window(normalized_window_type, num_points, fftbins=True)
    return window_array

def compute_fft_spectrum(
    signal_data: np.ndarray,
    sample_rate_hz: float,
    center_frequency_hz: float = 0.0,
    window_type: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT spectrum in dB.

    Args:
        signal_data: 1D complex baseband signal.
        sample_rate_hz: Sampling rate in Hz.
        center_frequency_hz: Optional center frequency offset in Hz.
        window_type: FFT window type.

    Returns:
        Tuple:
        - frequency_axis_hz: Frequency axis in Hz
        - spectrum_db: Magnitude spectrum in dB
    """
    if signal_data.ndim != 1:
        raise ValueError("signal_data must be a 1D signal.")

    num_samples = len(signal_data)
    if num_samples < 1:
        raise ValueError("signal_data must not be empty.")

    window_values = _resolve_window(window_type, num_samples)
    windowed_signal = signal_data * window_values

    fft_values = np.fft.fftshift(np.fft.fft(windowed_signal))
    frequency_axis_hz = (
        np.fft.fftshift(np.fft.fftfreq(num_samples, d=1.0 / sample_rate_hz))
        + center_frequency_hz
    )

    magnitude_values = np.abs(fft_values)
    spectrum_db = 20.0 * np.log10(np.maximum(magnitude_values, 1e-12))

    return frequency_axis_hz, spectrum_db


def compute_spectrum_metrics(
    frequency_axis_hz: np.ndarray,
    spectrum_db: np.ndarray,
) -> dict:
    """
    Compute summary metrics from an FFT spectrum.

    Returns:
        Dictionary with:
        - peak_frequency_hz
        - peak_power_db
        - mean_level_db
    """
    if frequency_axis_hz.ndim != 1 or spectrum_db.ndim != 1:
        raise ValueError("frequency_axis_hz and spectrum_db must be 1D arrays.")

    if len(frequency_axis_hz) != len(spectrum_db):
        raise ValueError("frequency_axis_hz and spectrum_db must have the same length.")

    peak_index = int(np.argmax(spectrum_db))

    return {
        "peak_frequency_hz": float(frequency_axis_hz[peak_index]),
        "peak_power_db": float(spectrum_db[peak_index]),
        "mean_level_db": float(np.mean(spectrum_db)),
    }


def compute_spectrogram_data(
    signal_data: np.ndarray,
    sample_rate_hz: float,
    center_frequency_hz: float = 0.0,
    window_type: str = "hann",
    segment_length: int = 256,
    overlap_length: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram in dB.

    Args:
        signal_data: 1D complex signal.
        sample_rate_hz: Sampling rate in Hz.
        center_frequency_hz: Optional center frequency offset in Hz.
        window_type: Spectrogram window type.
        segment_length: Spectrogram window length.
        overlap_length: Overlap between windows.

    Returns:
        Tuple:
        - frequency_axis_hz
        - time_axis_s
        - spectrogram_db
    """
    if signal_data.ndim != 1:
        raise ValueError("signal_data must be a 1D signal.")

    if segment_length < 2:
        raise ValueError("segment_length must be at least 2.")

    if overlap_length < 0 or overlap_length >= segment_length:
        raise ValueError("overlap_length must satisfy 0 <= overlap_length < segment_length.")

    window_values = _resolve_window(window_type, segment_length)

    frequency_axis_hz, time_axis_s, spectrogram_magnitude = spectrogram(
        signal_data,
        fs=sample_rate_hz,
        window=window_values,
        nperseg=segment_length,
        noverlap=overlap_length,
        mode="magnitude",
        return_onesided=False,
    )

    frequency_axis_hz = np.fft.fftshift(frequency_axis_hz) + center_frequency_hz
    spectrogram_magnitude = np.fft.fftshift(spectrogram_magnitude, axes=0)
    spectrogram_db = 20.0 * np.log10(np.maximum(spectrogram_magnitude, 1e-12))

    return frequency_axis_hz, time_axis_s, spectrogram_db