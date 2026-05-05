"""
RFDetect Lab - Analysis Pipeline

UI-independent end-to-end DSP pipeline for RF bearing estimation.
This module allows the analysis chain to run from Streamlit, CLI,
batch tests, or future C++ benchmarking.
"""

import time
import numpy as np

from config import ANGLE_GRID_DEG, ANTENNA_SPACING_M, WAVELENGTH_M

from src.signal_generator import (
    generate_tone_signal,
    generate_burst_signal,
    generate_array_data,
    generate_array_data_multi,
)

from src.preprocessing import (
    remove_dc,
    normalize_signal,
    smoothing_signal,
)

from src.detection import (
    detect_signal_energy,
    ca_cfar_1d,
)

from src.beamforming import (
    beamform_response,
    compute_sample_covariance,
    find_spectrum_peaks,
    peak_extraction_from_spectrum,
    music_spectrum,
    process_bearings_over_time,
    estimate_bearing_from_phase_difference,
)

from src.metrics import (
    compute_peak_power,
    compute_peak_to_average_ratio,
    compute_peak_to_sidelobe_ratio,
)

from src.spectrum_analysis import (
    compute_fft_spectrum,
    compute_spectrum_metrics,
    compute_spectrogram_data,
)


def make_source_signal(
    selected_signal_type: str,
    frequency_hz: float,
    num_samples_local: int,
) -> np.ndarray:
    if selected_signal_type == "tone":
        return generate_tone_signal(
            frequency_hz=frequency_hz,
            sample_rate_hz=1e6,
            num_samples=num_samples_local,
        )

    if selected_signal_type == "burst":
        burst_start = num_samples_local // 4
        burst_length = num_samples_local // 2
        return generate_burst_signal(
            frequency_hz=frequency_hz,
            sample_rate_hz=1e6,
            num_samples=num_samples_local,
            burst_start=burst_start,
            burst_length=burst_length,
        )

    return np.zeros(num_samples_local, dtype=np.complex128)


def compute_emitter_resolution_diagnostics(
    true_angles_deg,
    num_antennas,
    antenna_spacing_m,
    wavelength_m,
):
    """
    Estimate whether the configured emitters are likely resolvable
    by conventional beamforming.

    Uses an approximate ULA half-power beamwidth near broadside:

        HPBW ≈ 0.886 * wavelength / (M * d)

    """
    if true_angles_deg is None or len(true_angles_deg) < 2:
        return {
            "enabled": False,
            "message": "Single-emitter scenario: resolution diagnostics not required.",
            "min_separation_deg": None,
            "beamwidth_deg": None,
            "status": "OK",
        }

    sorted_angles = sorted(float(angle) for angle in true_angles_deg)
    separations = [
        abs(sorted_angles[i + 1] - sorted_angles[i])
        for i in range(len(sorted_angles) - 1)
    ]

    min_separation_deg = min(separations)

    beamwidth_rad = 0.886 * wavelength_m / (num_antennas * antenna_spacing_m)
    beamwidth_deg = float(np.rad2deg(beamwidth_rad))

    if min_separation_deg < beamwidth_deg:
        status = "Warning"
        message = "Emitter spacing is below the approximate beamforming resolution limit."
    else:
        status = "OK"
        message = "Emitter spacing is above the approximate beamforming resolution limit."

    return {
        "enabled": True,
        "message": message,
        "min_separation_deg": float(min_separation_deg),
        "beamwidth_deg": beamwidth_deg,
        "status": status,
    }


def compute_coherent_source_warning(
    signal_type,
    emitter_frequencies_hz,
    num_emitters,
):
    """
    Check whether multiple emitters may be coherent or highly correlated.
    
    """

    if num_emitters < 2 or signal_type == "noise_only":
        return {
            "enabled": False,
            "status": "OK",
            "message": "Coherent-source diagnostics not required for this scenario.",
            "min_frequency_spacing_hz": None,
        }

    sorted_freqs = sorted(float(freq) for freq in emitter_frequencies_hz)

    frequency_spacings = [
        abs(sorted_freqs[i + 1] - sorted_freqs[i])
        for i in range(len(sorted_freqs) - 1)
    ]

    min_frequency_spacing_hz = min(frequency_spacings)

    if min_frequency_spacing_hz == 0:
        status = "Warning"
        message = "Multiple emitters use the same frequency. Their signals may be coherent."
    elif min_frequency_spacing_hz < 5_000:
        status = "Caution"
        message = "Emitter frequencies are very close. The sources may be highly correlated."
    else:
        status = "OK"
        message = "Emitter frequencies are sufficiently separated."

    return {
        "enabled": True,
        "status": status,
        "message": message,
        "min_frequency_spacing_hz": float(min_frequency_spacing_hz),
    }


def compute_matched_mean_error(true_angles_deg, estimated_angles_deg):
    """
    Calculates the Mean Absolute Error (MAE) between ground truth DOA angles 
    and estimates using a greedy nearest-neighbor matching strategy.

    Args:
        true_angles_deg (array-like): The actual angles of arrival in degrees.
        estimated_angles_deg (array-like): The angles predicted by the DOA 
            algorithm 

    Returns:
        float: The average angular error of the matched pairs. 
   
    """
    if true_angles_deg is None or estimated_angles_deg is None or len(estimated_angles_deg) == 0:
        return None

    remaining_estimates = list(estimated_angles_deg)
    matched_errors = []

    for true_angle in true_angles_deg:
        if not remaining_estimates:
            break

        distances = [abs(true_angle - est) for est in remaining_estimates]
        best_idx = int(np.argmin(distances))
        matched_errors.append(distances[best_idx])
        remaining_estimates.pop(best_idx)

    return float(np.mean(matched_errors)) if matched_errors else None


def normalize_spatial_spectrum(spatial_spectrum):
    """
    Normalizes a spatial pseudospectrum by scaling its values relative to the maximum peak.

    Args:
        spatial_spectrum (array-like): The raw spatial spectrum values 1D NumPy array

    Returns:
        array-like: The normalized spectrum where the maximum value is 1.0. 
            
    """
    if spatial_spectrum is None or len(spatial_spectrum) == 0:
        return spatial_spectrum

    spectrum_max = float(np.max(spatial_spectrum))
    if spectrum_max <= 0.0:
        return spatial_spectrum

    return spatial_spectrum / spectrum_max


def run_aoa_method_for_comparison(
    method_name,
    processed_array_data,
    true_angles_for_display,
    num_sources,
    peak_mode,
    min_peak_separation_deg,
    max_reported_peaks,
    peak_threshold,
    smoothing,
    num_antennas,
):
    start_time = time.perf_counter()
    phase_difference_rad = None

    try:
        if method_name == "Beamforming":
            spatial_spectrum = beamform_response(
                array_data=processed_array_data,
                angle_grid_deg=ANGLE_GRID_DEG,
                num_antennas=num_antennas,
            )

        elif method_name == "MUSIC":
            covariance_matrix = compute_sample_covariance(processed_array_data)
            method_input_data = smoothing_signal(covariance_matrix) if smoothing else covariance_matrix

            safe_num_sources = int(np.clip(num_sources, 1, num_antennas - 1))

            spatial_spectrum = music_spectrum(
                array_data=method_input_data,
                num_sources=safe_num_sources,
                angle_grid_deg=ANGLE_GRID_DEG,
                num_antennas=num_antennas,
            )

        elif method_name == "Phase-Difference":
            estimated_angle_deg, phase_difference_rad, pair_angles_deg = (
                estimate_bearing_from_phase_difference(processed_array_data)
            )

            spatial_spectrum = np.zeros_like(ANGLE_GRID_DEG, dtype=np.float64)
            nearest_idx = int(np.argmin(np.abs(ANGLE_GRID_DEG - estimated_angle_deg)))
            spatial_spectrum[nearest_idx] = 1.0

            detected_angles_deg = np.array([estimated_angle_deg], dtype=np.float64)
            detected_peak_values = np.array([1.0], dtype=np.float64)

        else:
            raise ValueError(f"Unsupported AoA method: {method_name}")

        if method_name != "Phase-Difference":
            if peak_mode == "Top-N Peaks":
                detected_angles_deg, detected_peak_values = find_spectrum_peaks(
                    angle_grid_deg=ANGLE_GRID_DEG,
                    spatial_spectrum=spatial_spectrum,
                    min_separation_deg=float(min_peak_separation_deg),
                    max_peaks=max_reported_peaks,
                    min_peak_height=None,
                )
            else:
                detected_angles_deg, detected_peak_values = peak_extraction_from_spectrum(
                    spatial_spectrum=spatial_spectrum,
                    angles=ANGLE_GRID_DEG,
                    p_threshold=peak_threshold,
                )

        runtime_ms = (time.perf_counter() - start_time) * 1000.0

        display_spectrum = normalize_spatial_spectrum(spatial_spectrum.copy())
        par_value = compute_peak_to_average_ratio(display_spectrum)
        pslr_value = compute_peak_to_sidelobe_ratio(display_spectrum)

        raw_confidence = float(par_value * pslr_value)
        confidence_value = 100.0 * (1.0 - np.exp(-0.5 * raw_confidence))
        confidence_value = min(confidence_value, 100.0)

        mean_error = compute_matched_mean_error(
            true_angles_for_display,
            detected_angles_deg,
        )

        return {
            "method": method_name,
            "spatial_spectrum": spatial_spectrum,
            "detected_angles_deg": detected_angles_deg,
            "detected_peak_values": detected_peak_values,
            "mean_error_deg": mean_error,
            "runtime_ms": runtime_ms,
            "confidence_score": confidence_value,
            "par": par_value,
            "pslr": pslr_value,
            "phase_difference_rad": phase_difference_rad,
            "status": "OK",
            "notes": (
                "Single dominant-source assumption"
                if method_name == "Phase-Difference"
                else "Spectrum-based estimator"
            ),
        }

    except Exception as exc:
        runtime_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "method": method_name,
            "spatial_spectrum": None,
            "detected_angles_deg": np.array([], dtype=np.float64),
            "detected_peak_values": np.array([], dtype=np.float64),
            "mean_error_deg": None,
            "runtime_ms": runtime_ms,
            "confidence_score": None,
            "par": None,
            "pslr": None,
            "phase_difference_rad": None,
            "status": "Error",
            "notes": str(exc),
        }


def run_analysis_pipeline(cfg: dict) -> dict:
    signal_type = cfg["signal_type"]
    num_emitters = cfg["num_emitters"]
    emitter_angles_deg = cfg["emitter_angles_deg"]
    emitter_amplitudes = cfg["emitter_amplitudes"]
    emitter_frequencies_hz = cfg["emitter_frequencies_hz"]

    snr_db = cfg["snr_db"]
    num_samples = cfg["num_samples"]
    num_antennas = cfg["num_antennas"]

    detection_type = cfg["detection_type"]
    detection_threshold = cfg["detection_threshold"]
    cfar_detection_scale = cfg["cfar_detection_scale"]

    estimation_method = cfg["estimation_method"]
    num_sources = cfg["num_sources"]
    peak_mode = cfg["peak_mode"]
    min_peak_separation_deg = cfg["min_peak_separation_deg"]
    max_reported_peaks = cfg["max_reported_peaks"]
    peak_threshold = cfg["peak_threshold"]
    frame_length = cfg["frame_length"]
    smoothing = cfg["smoothing"]

    spectrum_window_type = cfg["spectrum_window_type"]
    spectrogram_segment_length = cfg["spectrogram_segment_length"]
    spectrogram_overlap_length = cfg["spectrogram_overlap_length"]

    source_signals = [
        make_source_signal(
            selected_signal_type=signal_type,
            frequency_hz=freq_hz,
            num_samples_local=num_samples,
        )
        for freq_hz in emitter_frequencies_hz
    ]

    emitters = [
        {
            "source_signal": source_signals[idx],
            "angle_deg": emitter_angles_deg[idx],
            "amplitude": emitter_amplitudes[idx],
        }
        for idx in range(num_emitters)
    ]

    if num_emitters == 1:
        array_data = generate_array_data(
            source_signal=source_signals[0],
            angle_deg=emitter_angles_deg[0],
            snr_db=snr_db,
            num_antennas=num_antennas,
        )
    else:
        array_data = generate_array_data_multi(
            emitters=emitters,
            snr_db=snr_db,
            num_antennas=num_antennas,
        )

    true_angles_for_display = emitter_angles_deg if signal_type != "noise_only" else None

    resolution_diagnostics = compute_emitter_resolution_diagnostics(
        true_angles_deg=true_angles_for_display,
        num_antennas=num_antennas,
        antenna_spacing_m=ANTENNA_SPACING_M,
        wavelength_m=WAVELENGTH_M,
    )

    coherent_source_diagnostics = compute_coherent_source_warning(
        signal_type=signal_type,
        emitter_frequencies_hz=emitter_frequencies_hz,
        num_emitters=num_emitters,
    )

    processed_array_data = remove_dc(array_data)
    processed_array_data = normalize_signal(processed_array_data)

    sig_detected, sig_energy, instantaneous_power = detect_signal_energy(
        processed_array_data[0],
        threshold=detection_threshold,
    )

    sample_cfar_detections, sample_cfar_thresholds = ca_cfar_1d(
        instant_power=instantaneous_power,
        num_training_cells=12,
        num_guard_cells=2,
        threshold_scale=cfar_detection_scale,
    )

    cfar_detected = bool(np.any(sample_cfar_detections))
    cfar_hit_indices = np.flatnonzero(sample_cfar_detections)
    cfar_energy = (
        float(np.mean(instantaneous_power[cfar_hit_indices]))
        if len(cfar_hit_indices) > 0
        else 0.0
    )

    if detection_type == "Fixed threshold":
        detected = sig_detected
        energy = sig_energy
    else:
        detected = cfar_detected
        energy = cfar_energy

    analysis_signal = processed_array_data[0]
    sample_indices = np.arange(len(analysis_signal))

    fft_freqs_hz, fft_spectrum_db = compute_fft_spectrum(
        signal_data=analysis_signal,
        sample_rate_hz=1e6,
        center_frequency_hz=0.0,
    )

    spectrum_metrics = compute_spectrum_metrics(
        frequency_axis_hz=fft_freqs_hz,
        spectrum_db=fft_spectrum_db,
    )

    effective_overlap_length = min(
        spectrogram_overlap_length,
        spectrogram_segment_length - 1,
    )

    effective_segment_length = min(
        spectrogram_segment_length,
        len(analysis_signal),
    )

    spectrogram_frequency_axis_hz, spectrogram_time_axis_s, spectrogram_db = (
        compute_spectrogram_data(
            signal_data=analysis_signal,
            sample_rate_hz=1e6,
            center_frequency_hz=0.0,
            window_type=spectrum_window_type,
            segment_length=effective_segment_length,
            overlap_length=effective_overlap_length,
        )
    )

    if detected:
        aoa_result = run_aoa_method_for_comparison(
            method_name=estimation_method,
            processed_array_data=processed_array_data,
            true_angles_for_display=true_angles_for_display,
            num_sources=num_sources,
            peak_mode=peak_mode,
            min_peak_separation_deg=min_peak_separation_deg,
            max_reported_peaks=max_reported_peaks,
            peak_threshold=peak_threshold,
            smoothing=smoothing,
            num_antennas=num_antennas,
        )

        spatial_spectrum = aoa_result["spatial_spectrum"]
        detected_angles_deg = aoa_result["detected_angles_deg"]
        detected_peak_values = aoa_result["detected_peak_values"]
        mean_angle_error = aoa_result["mean_error_deg"]
        confidence_score = aoa_result["confidence_score"]
        par = aoa_result["par"]
        pslr = aoa_result["pslr"]
        phase_difference_rad = aoa_result["phase_difference_rad"]

        display_spectrum = (
            normalize_spatial_spectrum(spatial_spectrum.copy())
            if spatial_spectrum is not None
            else None
        )

        peak_power = (
            compute_peak_power(display_spectrum)
            if display_spectrum is not None
            else None
        )

        if estimation_method != "Phase-Difference":
            frame_indices, frame_detected_angles_deg, frame_detected_peak_values = (
                process_bearings_over_time(
                    array_data=processed_array_data,
                    angle_grid_deg=ANGLE_GRID_DEG,
                    frame_length=frame_length,
                    estimation_method=estimation_method,
                    num_sources=num_sources,
                    peak_mode=peak_mode,
                    min_peak_separation_deg=float(min_peak_separation_deg),
                    max_reported_peaks=max_reported_peaks,
                    smoothing=smoothing,
                    p_threshold=peak_threshold,
                    num_antennas=num_antennas,
                )
            )
        else:
            frame_indices = np.array([], dtype=np.int64)
            frame_detected_angles_deg = []
            frame_detected_peak_values = []

    else:
        aoa_result = None
        spatial_spectrum = None
        detected_angles_deg = np.array([], dtype=np.float64)
        detected_peak_values = np.array([], dtype=np.float64)
        mean_angle_error = None
        confidence_score = None
        par = None
        pslr = None
        phase_difference_rad = None
        peak_power = None
        frame_indices = np.array([], dtype=np.int64)
        frame_detected_angles_deg = []
        frame_detected_peak_values = []

    is_reliable = (
        detected
        and confidence_score is not None
        and confidence_score >= cfg["confidence_threshold"]
        and par is not None
        and par >= 3.0
    )

    return {
        "array_data": array_data,
        "processed_array_data": processed_array_data,

        "detected": detected,
        "sig_detected": sig_detected,
        "cfar_detected": cfar_detected,
        "energy": energy,
        "sig_energy": sig_energy,
        "cfar_energy": cfar_energy,
        "instantaneous_power": instantaneous_power,
        "sample_cfar_detections": sample_cfar_detections,
        "sample_cfar_thresholds": sample_cfar_thresholds,
        "sample_indices": sample_indices,

        "fft_freqs_hz": fft_freqs_hz,
        "fft_spectrum_db": fft_spectrum_db,
        "spectrogram_frequency_axis_hz": spectrogram_frequency_axis_hz,
        "spectrogram_time_axis_s": spectrogram_time_axis_s,
        "spectrogram_db": spectrogram_db,
        "spectrum_metrics": spectrum_metrics,

        "aoa_result": aoa_result,
        "spatial_spectrum": spatial_spectrum,
        "detected_angles_deg": detected_angles_deg,
        "detected_peak_values": detected_peak_values,
        "mean_angle_error": mean_angle_error,
        "confidence_score": confidence_score,
        "par": par,
        "pslr": pslr,
        "phase_difference_rad": phase_difference_rad,
        "peak_power": peak_power,

        "frame_indices": frame_indices,
        "frame_detected_angles_deg": frame_detected_angles_deg,
        "frame_detected_peak_values": frame_detected_peak_values,

        "true_angles_for_display": true_angles_for_display,
        "resolution_diagnostics": resolution_diagnostics,
        "coherent_source_diagnostics": coherent_source_diagnostics,
        "is_reliable": is_reliable,
    }