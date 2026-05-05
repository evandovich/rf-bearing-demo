"""
RFDetect Lab - Signal Processing Module

Author: Evans Baidoo
Copyright (c) 2026 Evans Baidoo

Beamforming and angle-of-arrival (AoA) estimation.

"""

import numpy as np

from config import ANGLE_GRID_DEG, ANTENNA_SPACING_M, NUM_ANTENNAS, WAVELENGTH_M
from src.signal_generator import steering_vector_ula
from src.detection import ca_cfar_1d, find_cfar_confirmed_peaks
from src.preprocessing import smoothing_signal

def beamform_response(
    array_data: np.ndarray,
    angle_grid_deg: np.ndarray = ANGLE_GRID_DEG,
    num_antennas: int = NUM_ANTENNAS,
    antenna_spacing_m: float = ANTENNA_SPACING_M,
    wavelength_m: float = WAVELENGTH_M,
) -> np.ndarray:
    """
    Compute conventional beamformer spatial response over an angle grid.

    Args:
        array_data: Array data of shape (num_antennas, num_samples).
        angle_grid_deg: Angles to scan in degrees.
        num_antennas: Number of antennas.
        antenna_spacing_m: Antenna spacing in meters.
        wavelength_m: Wavelength in meters.

    Returns:
        Spatial spectrum of shape (len(angle_grid_deg),).
    """
    if array_data.ndim != 2:
        raise ValueError("Input array_data must have shape (num_antennas, num_samples).")

    if array_data.shape[0] != num_antennas:
        raise ValueError("First dimension of array_data must match num_antennas.")

    spatial_spectrum = np.zeros(len(angle_grid_deg), dtype=np.float64)

    # for i, angle_deg in enumerate(angle_grid_deg):
    #     steer_vec = steering_vector_ula(
    #         angle_deg=angle_deg,
    #         num_antennas=num_antennas,
    #         antenna_spacing_m=antenna_spacing_m,
    #         wavelength_m=wavelength_m,
    #     )

    #     beam_weights = steer_vec / num_antennas
    #     beam_output = np.conjugate(beam_weights) @ array_data
    #     spatial_spectrum[i] = np.mean(np.abs(beam_output) ** 2)
    steering_matrix = np.column_stack([
        steering_vector_ula(
            angle_deg=angle_deg,
            num_antennas=num_antennas,
            antenna_spacing_m=antenna_spacing_m,
            wavelength_m=wavelength_m,
        )
        for angle_deg in angle_grid_deg
    ])

    beam_weights = steering_matrix / num_antennas

    beam_outputs = np.conjugate(beam_weights).T @ array_data

    spatial_spectrum = np.mean(np.abs(beam_outputs) ** 2, axis=1)

    return spatial_spectrum


def compute_sample_covariance(array_data: np.ndarray) -> np.ndarray:
    """
    Compute the sample covariance matrix of array data.

    Args:
        array_data: Array data of shape (num_antennas, num_samples).

    Returns:
        Covariance matrix of shape (num_antennas, num_antennas).
    """
    if array_data.ndim != 2:
        raise ValueError("array_data must have shape (num_antennas, num_samples).")

    num_antennas, num_samples = array_data.shape

    if num_samples < 1:
        raise ValueError("array_data must contain at least one sample.")

    covariance = (array_data @ np.conjugate(array_data.T)) / num_samples
    return covariance


def get_noise_subspace(
    covariance_matrix: np.ndarray,
    num_sources: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues/eigenvectors and return the noise subspace.

    Args:
        covariance_matrix: Covariance matrix of shape (M, M).
        num_sources: Number of signal sources.

    Returns:
        Tuple:
        - eigenvalues_desc: Eigenvalues sorted descending
        - noise_subspace: Noise-subspace eigenvectors of shape (M, M - num_sources)
    """
    if covariance_matrix.ndim != 2:
        raise ValueError("covariance_matrix must be 2D.")

    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("covariance_matrix must be square.")

    num_antennas = covariance_matrix.shape[0]

    if not (0 <= num_sources < num_antennas):
        raise ValueError("num_sources must satisfy 0 <= num_sources < num_antennas.")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_desc = eigenvalues[sort_idx]
    eigenvectors_desc = eigenvectors[:, sort_idx]

    noise_subspace = eigenvectors_desc[:, num_sources:]

    return eigenvalues_desc, noise_subspace


def music_spectrum(
    array_data: np.ndarray,
    num_sources: int,
    angle_grid_deg: np.ndarray = ANGLE_GRID_DEG,
    num_antennas: int = NUM_ANTENNAS,
    antenna_spacing_m: float = ANTENNA_SPACING_M,
    wavelength_m: float = WAVELENGTH_M,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Compute the MUSIC pseudospectrum over an angle grid.

    Args:
        array_data: Array data of shape (num_antennas, num_samples).
        num_sources: Number of signal sources.
        angle_grid_deg: Angles to scan in degrees.
        num_antennas: Number of antennas.
        antenna_spacing_m: Antenna spacing in meters.
        wavelength_m: Wavelength in meters.
        epsilon: Small stabilizer to avoid division by zero.

    Returns:
        MUSIC pseudospectrum of shape (len(angle_grid_deg),).
    """
    if array_data.ndim != 2:
        raise ValueError("array_data must have shape (num_antennas, num_samples).")

    if array_data.shape[0] != num_antennas:
        raise ValueError("First dimension of array_data must match num_antennas.")

    # covariance_matrix = compute_sample_covariance(array_data)
    _, noise_subspace = get_noise_subspace(array_data, num_sources=num_sources)

    pseudospectrum = np.zeros(len(angle_grid_deg), dtype=np.float64)

    for i, angle_deg in enumerate(angle_grid_deg):
        steering = steering_vector_ula(
            angle_deg=angle_deg,
            num_antennas=num_antennas,
            antenna_spacing_m=antenna_spacing_m,
            wavelength_m=wavelength_m,
        )[:, np.newaxis]

        denominator = np.conjugate(steering).T @ noise_subspace @ np.conjugate(noise_subspace).T @ steering
        denominator_val = float(np.real(denominator.item()))

        pseudospectrum[i] = 1.0 / max(denominator_val, epsilon)

    return pseudospectrum


def estimate_bearing_from_phase_difference(
    array_data: np.ndarray,
    antenna_spacing_m: float = ANTENNA_SPACING_M,
    wavelength_m: float = WAVELENGTH_M,
) -> tuple[float, float,  np.ndarray]:
    """
    Estimate AoA from average phase difference across adjacent antenna pairs.

    This method assumes a single dominant narrowband source and a ULA.

    Args:
        array_data: Array data of shape (num_antennas, num_samples).
        antenna_spacing_m: Antenna spacing in meters.
        wavelength_m: Wavelength in meters.

    Returns:
        Tuple:
        - estimated_angle_deg: Estimated bearing in degrees
        - mean_phase_diff_rad: Mean adjacent-pair phase difference in radians
    """
    if array_data.ndim != 2:
        raise ValueError("array_data must have shape (num_antennas, num_samples).")

    num_antennas, num_samples = array_data.shape

    if num_antennas < 2:
        raise ValueError("At least two antennas are required for phase-difference AoA.")

    if num_samples < 1:
        raise ValueError("array_data must contain at least one sample.")

    phase_diffs = []

    for m in range(num_antennas - 1):
        # Cross-correlation between adjacent channels
        cross = np.mean(array_data[m + 1] * np.conjugate(array_data[m]))
        phase_diff = np.angle(cross)
        phase_diffs.append(phase_diff)

    # Constant for converting phase to the sine of the angle
    sensitivity_factor = -wavelength_m / (2.0 * np.pi * antenna_spacing_m)

    # 1. Individual antenna check (Sanity/Ambiguity)
    ant_sin_theta = np.clip(np.array(phase_diffs) * sensitivity_factor, -1.0, 1.0)
    ant_est_angle_deg = np.rad2deg(np.arcsin(ant_sin_theta))

    # 2. Mean phase estimator 
    mean_phase_diff_rad = float(np.mean(phase_diffs))
    mean_sin_theta = np.clip(np.mean(mean_phase_diff_rad) * sensitivity_factor, -1.0, 1.0)
    estimated_angle_deg = float(np.rad2deg(np.arcsin(mean_sin_theta)))


    return estimated_angle_deg, mean_phase_diff_rad, ant_est_angle_deg


def estimate_bearing_from_spectrum(
    angle_grid_deg: np.ndarray,
    spatial_spectrum: np.ndarray,
) -> tuple[float, float]:
    """
    Estimate bearing from a spatial spectrum by selecting the peak angle.

    Args:
        angle_grid_deg: Angle scan grid in degrees.
        spatial_spectrum: Spectrum response for each angle.

    Returns:
        Tuple:
        - estimated_angle_deg: Angle corresponding to the maximum response
        - peak_value: Maximum spectrum value
    """
    if angle_grid_deg.ndim != 1:
        raise ValueError("angle_grid_deg must be a 1D array.")

    if spatial_spectrum.ndim != 1:
        raise ValueError("spatial_spectrum must be a 1D array.")

    if len(angle_grid_deg) != len(spatial_spectrum):
        raise ValueError("angle_grid_deg and spatial_spectrum must have same length.")

    peak_index = int(np.argmax(spatial_spectrum))
    estimated_angle_deg = float(angle_grid_deg[peak_index])
    peak_value = float(spatial_spectrum[peak_index])

    return estimated_angle_deg, peak_value


def find_spectrum_peaks(
    angle_grid_deg: np.ndarray,
    spatial_spectrum: np.ndarray,
    min_separation_deg: float = 5.0,
    max_peaks: int = 1,
    min_peak_height: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima in a spectrum and suppress peaks that are too close together.

    Args:
        angle_grid_deg: 1D angle grid in degrees.
        spatial_spectrum: 1D spectrum values.
        min_separation_deg: Minimum separation between returned peaks in degrees.
        max_peaks: Maximum number of peaks to return. If None, return all valid peaks.
        min_peak_height: Minimum required peak height. If None, no amplitude threshold is applied.

    Returns:
          Tuple:
         - estimated_angles_deg: 1D array of estimated angles
         - peak_values: 1D array of corresponding peak values
    """
    if angle_grid_deg.ndim != 1:
        raise ValueError("angle_grid_deg must be a 1D array.")

    if spatial_spectrum.ndim != 1:
        raise ValueError("spatial_spectrum must be a 1D array.")

    if len(angle_grid_deg) != len(spatial_spectrum):
        raise ValueError("angle_grid_deg and spatial_spectrum must have same length.")

    if len(spatial_spectrum) < 3:
        return np.array([], dtype=int)

    norm_spectrum = np.asarray(spatial_spectrum/np.max(spatial_spectrum))

    # Find local maxima indices on the  grid
    candidate_indices = np.where(np.diff(np.sign(np.diff(norm_spectrum))) < 0)[0] + 1
    sorted_candidate_indices = candidate_indices[np.argsort(norm_spectrum[candidate_indices])[::-1]]
    selected_indices = sorted_candidate_indices[: max_peaks]

    for idx in selected_indices:
        angle_deg = angle_grid_deg[idx]

        too_close = any(
            abs(angle_deg - angle_grid_deg[selected_idx]) < min_separation_deg
            for selected_idx in selected_indices
        )

        if not too_close:
            selected_indices.append(int(idx))
    
    estimated_angles_deg = angle_grid_deg[selected_indices].astype(np.float64)
    peak_values = norm_spectrum[selected_indices].astype(np.float64)

    return estimated_angles_deg, peak_values


def peak_extraction_from_spectrum(spatial_spectrum, angles, p_threshold):
    """
    Generic peak detection for Normalized MUSIC Pseudospectrum or Beamforming.
    
    Args:
        spatial_spectrum: 1D array of normalized values [0, 1]
        angles: 1D array of corresponding angles (e.g., -90 to 90)
        p_threshold: Sensitivity threshold (default 0.3)
        
    Returns:
        detected_angles: List of angles where local maxima occur above threshold
        det_peak_pwrs: list of peaks values where local maxima occur above threshold
    """
    norm_spectrum = np.asarray(spatial_spectrum/np.max(spatial_spectrum))
        
    # Find local maxima indices on the  grid
    candidate_indices = np.where(np.diff(np.sign(np.diff(norm_spectrum))) < 0)[0] + 1
    sorted_candidate_indices = candidate_indices[np.argsort(norm_spectrum[candidate_indices])[::-1]]
    local_max_idxs = sorted_candidate_indices[norm_spectrum[sorted_candidate_indices] > p_threshold]

    detected_angles = angles[local_max_idxs].astype(np.float64)
    det_peak_pwr = norm_spectrum[local_max_idxs].astype(np.float64)        
    
    return detected_angles, det_peak_pwr


# def estimate_bearings_from_spectrum(
#     angle_grid_deg: np.ndarray,
#     spatial_spectrum: np.ndarray,
#     min_separation_deg: float = 5.0,
#     max_peaks: int | None = None,
#     min_peak_height: float | None = None,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Estimate multiple bearings from a spectrum using local peak detection.

#     Args:
#         angle_grid_deg: 1D angle grid in degrees.
#         spatial_spectrum: 1D spectrum values.
#         min_separation_deg: Minimum angular separation between returned peaks.
#         max_peaks: Maximum number of peaks to return. If None, return all valid peaks.
#         min_peak_height: Minimum peak height required for a detection.

#     Returns:
#         Tuple:
#         - estimated_angles_deg: 1D array of estimated angles
#         - peak_values: 1D array of corresponding peak values
#     """
#     peak_indices = find_spectrum_peaks(
#         angle_grid_deg=angle_grid_deg,
#         spatial_spectrum=spatial_spectrum,
#         min_separation_deg=min_separation_deg,
#         max_peaks=max_peaks,
#         min_peak_height=min_peak_height,
#     )
#     """
#     Find local maxima in a spectrum and suppress peaks that are too close together.

#     Args:
#         angle_grid_deg: 1D angle grid in degrees.
#         spatial_spectrum: 1D spectrum values.
#         min_separation_deg: Minimum separation between returned peaks in degrees.
#         max_peaks: Maximum number of peaks to return. If None, return all valid peaks.
#         min_peak_height: Minimum required peak height. If None, no amplitude threshold is applied.

#     Returns:
#         1D numpy array of selected peak indices, sorted by descending peak height.
#     """
#     if angle_grid_deg.ndim != 1:
#         raise ValueError("angle_grid_deg must be a 1D array.")

#     if spatial_spectrum.ndim != 1:
#         raise ValueError("spatial_spectrum must be a 1D array.")

#     if len(angle_grid_deg) != len(spatial_spectrum):
#         raise ValueError("angle_grid_deg and spatial_spectrum must have same length.")

#     if len(spatial_spectrum) < 3:
#         return np.array([], dtype=int)

#     norm_spectrum = np.asarray(spatial_spectrum/np.max(spatial_spectrum))   

#     # Find local maxima indices on the  grid
#     candidate_indices = np.where(np.diff(np.sign(np.diff(norm_spectrum))) < 0)[0] + 1
#     sorted_candidate_indices = candidate_indices[np.argsort(norm_spectrum[candidate_indices])[::-1]]
#     selected_indices = sorted_candidate_indices[: max_peaks]

#     for idx in selected_indices:
#         angle_deg = angle_grid_deg[idx]

#         too_close = any(
#             abs(angle_deg - angle_grid_deg[selected_idx]) < min_separation_deg
#             for selected_idx in selected_indices
#         )

#         if not too_close:
#             selected_indices.append(int(idx))
       

#     return np.array(selected_indices, dtype=int)

    # estimated_angles_deg = angle_grid_deg[peak_indices].astype(np.float64)
    # peak_values = spatial_spectrum[peak_indices].astype(np.float64)

    # return estimated_angles_deg, peak_values


# def track_bearing_over_time(
#     array_data: np.ndarray,
#     angle_grid_deg: np.ndarray,
#     frame_length: int,
#     num_antennas: int = NUM_ANTENNAS,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Estimate bearing frame-by-frame over time using single-peak beamforming.

#     Args:
#         array_data: Array data of shape (num_antennas, num_samples).
#         angle_grid_deg: Angle scan grid in degrees.
#         frame_length: Number of samples per frame.

#     Returns:
#         Tuple:
#         - frame_indices: 1D array of frame indices
#         - estimated_angles_deg: 1D array of estimated angle per frame
#     """
#     if array_data.ndim != 2:
#         raise ValueError("array_data must have shape (num_antennas, num_samples).")

#     _, num_samples = array_data.shape

#     if frame_length <= 0:
#         raise ValueError("frame_length must be positive.")

#     num_frames = num_samples // frame_length
#     if num_frames == 0:
#         raise ValueError("frame_length is larger than available samples.")

#     estimated_angles = []

#     for frame_idx in range(num_frames):
#         start = frame_idx * frame_length
#         end = start + frame_length

#         frame_data = array_data[:, start:end]

#         spatial_spectrum = beamform_response(
#             array_data=frame_data,
#             angle_grid_deg=angle_grid_deg,
#             num_antennas=num_antennas,
#         )

#         estimated_angle_deg, _ = estimate_bearing_from_spectrum(
#             angle_grid_deg=angle_grid_deg,
#             spatial_spectrum=spatial_spectrum,
#         )

#         estimated_angles.append(estimated_angle_deg)

#     frame_indices = np.arange(num_frames)
#     estimated_angles_deg = np.array(estimated_angles, dtype=np.float64)

#     return frame_indices, estimated_angles_deg

def process_bearings_over_time(
    array_data: np.ndarray,
    angle_grid_deg: np.ndarray,
    frame_length: int,
    estimation_method: str = "Beamforming",
    num_sources: int = 1,
    peak_mode: str = "Top-N Peaks",
    min_peak_separation_deg: float = 5.0,
    max_reported_peaks: int = 2,
    smoothing: bool = True,
    p_threshold: float = 0.3,
    num_antennas: int = NUM_ANTENNAS,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Process array data frame-by-frame and return multiple bearing detections per frame.

    This is a frame-wise multi-target bearing detector, not a full tracker with
    target association or persistent IDs.

    Args:
        array_data: Array data of shape (num_antennas, num_samples).
        angle_grid_deg: Angle scan grid in degrees.
        frame_length: Number of samples per frame.
        estimation_method: "Beamforming" or "MUSIC".
        num_sources: Number of assumed sources for MUSIC.
        peak_mode: "Top-N Peaks" or "CA-CFAR".
        min_peak_separation_deg: Minimum separation between accepted peaks.
        max_reported_peaks: Maximum number of peaks to report per frame.
        cfar_training_cells: Number of CFAR training cells on each side.
        cfar_guard_cells: Number of CFAR guard cells on each side.
        cfar_threshold_scale: CFAR threshold multiplier.

    Returns:
        Tuple:
        - frame_indices: 1D array of frame indices
        - frame_angles_deg: list of 1D arrays, one per frame
        - frame_peak_values: list of 1D arrays, one per frame
    """
    if array_data.ndim != 2:
        raise ValueError("array_data must have shape (num_antennas, num_samples).")

    if frame_length <= 0:
        raise ValueError("frame_length must be positive.")

    _, num_samples = array_data.shape
    num_frames = num_samples // frame_length

    if num_frames == 0:
        raise ValueError("frame_length is larger than available samples.")

    frame_indices = np.arange(num_frames)
    frame_angles_deg: list[np.ndarray] = []
    frame_peak_values: list[np.ndarray] = []

    for frame_idx in range(num_frames):
        start = frame_idx * frame_length
        end = start + frame_length
        frame_data = array_data[:, start:end]

        if smoothing:
            covariance_matrix = compute_sample_covariance(frame_data)
            frame_data = smoothing_signal(covariance_matrix)
        else:
            frame_data = compute_sample_covariance(frame_data) if estimation_method == "MUSIC" else frame_data

        
        if estimation_method == "Beamforming":
            spectrum = beamform_response(
                array_data=frame_data,
                angle_grid_deg=angle_grid_deg,
                num_antennas=num_antennas,
            )
        elif estimation_method == "MUSIC":
            spectrum = music_spectrum(
                array_data=frame_data,
                num_sources=num_sources,
                angle_grid_deg=angle_grid_deg,
                num_antennas=num_antennas,
            )
        else:
            raise ValueError("estimation_method must be 'Beamforming' or 'MUSIC'.")
                

        if peak_mode == "Top-N Peaks":
            detected_angles_deg, detected_peak_values = find_spectrum_peaks(
                angle_grid_deg=angle_grid_deg,
                spatial_spectrum=spectrum,
                min_separation_deg=min_peak_separation_deg,
                max_peaks=max_reported_peaks,
                min_peak_height=None,
            )

        elif peak_mode == "Threshold-Based":
            detected_angles_deg, detected_peak_values = peak_extraction_from_spectrum(                
                spatial_spectrum=spectrum,
                angles=ANGLE_GRID_DEG,
                p_threshold=p_threshold,
            )
       
        else:
            raise ValueError("peak_setting_method must be 'Top-N Peaks', 'Threshold-Based'.")

        frame_angles_deg.append(detected_angles_deg)
        frame_peak_values.append(detected_peak_values)

    return frame_indices, frame_angles_deg, frame_peak_values