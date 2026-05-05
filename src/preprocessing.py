"""
RFDetect Lab - Signal Processing Module

Author: Evans Baidoo
Copyright (c) 2026 Evans Baidoo

Signal preprocessing (DC removal, normalization, filtering).
"""

import numpy as np


def smoothing_signal(covariance):
    """
    Restores covariance matrix rank for coherent/multipath signals.
    
    Args:
        covariance: Original (M x M) covariance matrix (e.g., 4x4)
    
    Returns:
        fb_smoothing: Averaged (sub_size x sub_size) covariance matrix
    
    """
    M = covariance.shape[0]
    J = np.fliplr(np.eye(M))
    # Average covariance with its flipped conjugate version
    fb_smoothing = 0.5 * (covariance + J @ np.conj(covariance) @ J)
    return fb_smoothing

    # def patial_smoothing(covariance):        
    # """
    # Restores covariance matrix rank for coherent/multipath signals.
    
    # Args:
    #     R: Original (M x M) covariance matrix (e.g., 4x4)
    #     sub_size: Size of sub-arrays (must be < M)
        
    # Returns:
    #     R_smooth: Averaged (sub_size x sub_size) covariance matrix
    # """
    # M = R.shape[0]
    # num_subarrays = M - sub_size + 1
    # R_smooth = np.zeros((sub_size, sub_size), dtype=complex)
    
    # # Average overlapping sub-matrices along the main diagonal
    # for i in range(num_subarrays):
    #     R_smooth += R[i : i + sub_size, i : i + sub_size]
    
    # return R_smooth / num_subarrays


def remove_dc(x: np.ndarray) -> np.ndarray:
    """
    Remove DC component from a signal.

    Supports:
    - 1D input of shape (num_samples,)
    - 2D input of shape (num_channels, num_samples)

    Args:
        x: Input signal array.

    Returns:
        Signal with mean removed.
    """
    if x.ndim == 1:
        return x - np.mean(x)

    if x.ndim == 2:
        return x - np.mean(x, axis=1, keepdims=True)

    raise ValueError("Input must be 1D or 2D.")


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """
    Normalize signal by its maximum magnitude.

    Supports:
    - 1D input of shape (num_samples,)
    - 2D input of shape (num_channels, num_samples)

    Args:
        x: Input signal array.

    Returns:
        Normalized signal array.
    """
    max_val = np.max(np.abs(x))

    if max_val == 0.0:
        return x.copy()

    return x / max_val