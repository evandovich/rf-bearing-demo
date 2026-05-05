"""
Confidence and quality metrics for estimation results.
"""

import numpy as np

def compute_peak_power(spatial_spectrum: np.ndarray) -> float:
    """
    Compute the maximum value of the spatial spectrum.
    """
    return float(np.max(spatial_spectrum))


def compute_peak_to_average_ratio(spatial_spectrum: np.ndarray) -> float:
    """
    Compute peak-to-average ratio (PAR). This indicates how dominant the peak is over noise
    High PAR = strong detection

    PAR = max / mean
    """
    mean_val = np.mean(spatial_spectrum)
    if mean_val == 0.0:
        return 0.0
    return float(np.max(spatial_spectrum) / mean_val)


def compute_peak_to_sidelobe_ratio(spatial_spectrum: np.ndarray) -> float:
    """
    Compute peak-to-sidelobe ratio (PSLR). This indicates how clearly one direction stands out
    High PSLR = clean, reliable estimate

    PSLR = peak / second-highest peak
    """
    if len(spatial_spectrum) < 2:
        return 0.0

    sorted_vals = np.sort(spatial_spectrum)
    peak = sorted_vals[-1]
    second_peak = sorted_vals[-2]

    if second_peak == 0.0:
        return float("inf")

    return float(peak / second_peak)

