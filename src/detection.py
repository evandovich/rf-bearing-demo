"""
Signal detection methods (energy-based detection and CFAR on 1D spectra).
"""

import numpy as np


def compute_sample_signal(x: np.ndarray) -> float:
    """
    Extract average signal energy.
    Extract sample-wise instantaneous power: |x[n]|^2.
    Supports 1D or 2D input.

    Args:
        x: Input signal array.

    Returns:
        Average energy as a float.
    """
    instantaneous_power = np.abs(x) ** 2
    energy = float(np.mean(np.abs(x) ** 2))
    return instantaneous_power, energy


def detect_signal_energy(x: np.ndarray, threshold: float) -> tuple[bool, float]:
    """
    Detect signal presence using an energy threshold.

    Args:
        x: Input signal array.
        threshold: Energy threshold.

    Returns:
        Tuple:
        - detected: True if energy >= threshold
        - energy: computed signal energy
    """
    
    instantaneous_power, energy = compute_sample_signal(x)
    detected = energy >= threshold
    return detected, energy, instantaneous_power


def ca_cfar_1d(
    instant_power: np.ndarray,
    num_training_cells: int,
    num_guard_cells: int,
    threshold_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply 1D Cell-Averaging CFAR to a real-valued input spectrum.

    Args:
        instant_power: 1D real-valued input array (e.g. spatial spectrum).
        num_training_cells: Number of training cells on each side.
        num_guard_cells: Number of guard cells on each side.
        threshold_scale: Multiplier applied to the estimated noise level.

    Returns:
        Tuple:
        - detections: Boolean array of same length as instant_power
        - thresholds: Float array of same length as instant_power

    Notes:
        Cells near the edges that cannot support a full CFAR window are left as:
        - detections = False
        - thresholds = np.nan
    """
    if instant_power.ndim != 1:
        raise ValueError("instant_power must be a 1D array.")

    if num_training_cells <= 0:
        raise ValueError("num_training_cells must be positive.")

    if num_guard_cells < 0:
        raise ValueError("num_guard_cells must be nonnegative.")

    if threshold_scale <= 0.0:
        raise ValueError("threshold_scale must be positive.")

    n = len(instant_power)
    detections = np.zeros(n, dtype=bool)
    thresholds = np.full(n, np.nan, dtype=np.float64)

    half_window = num_training_cells + num_guard_cells

    for idx in range(n):
        left_start = idx - half_window
        left_end = idx - num_guard_cells
        right_start = idx + num_guard_cells + 1
        right_end = idx + half_window + 1

        if left_start < 0 or right_end > n:
            continue

        left_training = instant_power[left_start:left_end]
        right_training = instant_power[right_start:right_end]
        training_cells = np.concatenate([left_training, right_training])

        if len(training_cells) == 0:
            continue

        noise_estimate = float(np.mean(training_cells))
        threshold = threshold_scale * noise_estimate

        thresholds[idx] = threshold
        detections[idx] = bool(instant_power[idx] >= threshold)

    return detections, thresholds


def find_cfar_confirmed_peaks(
    x: np.ndarray,
    detections: np.ndarray,
) -> np.ndarray:
    """
    Return indices that are both:
    - local maxima in x
    - marked as detections by CFAR

    Args:
        x: 1D spectrum array.
        detections: 1D boolean detection mask from CFAR.

    Returns:
        1D array of peak indices.
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")

    if detections.ndim != 1:
        raise ValueError("detections must be a 1D array.")

    if len(x) != len(detections):
        raise ValueError("x and detections must have the same length.")

    peak_indices = []

    for idx in range(1, len(x) - 1):
        if not detections[idx]:
            continue

        is_local_max = x[idx] > x[idx - 1] and x[idx] >= x[idx + 1]
        if is_local_max:
            peak_indices.append(idx)

    return np.array(peak_indices, dtype=int)