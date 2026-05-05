"""
Plotting utilities for signals and spatial spectra.
"""

import matplotlib.pyplot as plt
import numpy as np


def _normalize_angle_input(angle_or_angles):
    if angle_or_angles is None:
        return []

    if np.isscalar(angle_or_angles):
        return [float(angle_or_angles)]

    return [float(x) for x in angle_or_angles]


def plot_spatial_spectrum(
    angle_grid_deg: np.ndarray,
    spatial_spectrum: np.ndarray,
    true_angles_deg=None,
    estimated_angles_deg=None,
    estimation_method="MUSIC",
) -> plt.Figure:
    """
    Plot beamforming spatial spectrum with optional multiple true and estimated angles.

    Args:
        angle_grid_deg: Angle scan grid in degrees.
        spatial_spectrum: Beamformer response.
        true_angles_deg: Optional scalar or iterable of true angles.
        estimated_angles_deg: Optional scalar or iterable of estimated angles.

    Returns:
        Matplotlib figure.
    """
    true_angles = _normalize_angle_input(true_angles_deg)
    estimated_angles = _normalize_angle_input(estimated_angles_deg)
    norm_spec = spatial_spectrum/max(spatial_spectrum)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(angle_grid_deg, norm_spec, label="Spatial Spectrum")

    for idx, angle_deg in enumerate(true_angles):
        ax.axvline(
            angle_deg,
            color='blue',
            linestyle="--",
            label="True Angle" if idx == 0 else None,
        )

    for idx, angle_deg in enumerate(estimated_angles):
        ax.axvline(
            angle_deg,
            color='orange',
            linestyle=":",
            label="Estimated Angle" if idx == 0 else None,
        )

    
    ax.set_xlabel("Angle (degrees)")
    if estimation_method == "MUSIC":
        ax.set_title("MUSIC AoA Estimation")
        ax.set_ylabel("Nomalized Pseudospectrum")
    elif estimation_method == "Phase-Difference":
        ax.set_title("Phase-Difference")
        ax.set_ylabel("Nomalized spectrum Power")
    else:
        ax.set_title("Conventional Beamforming")
        ax.set_ylabel("Nomalized spectrum Power")
    ax.legend(loc='best')
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_channel_real_part(array_data: np.ndarray) -> plt.Figure:
    """
    Plot the real part of each array channel.

    Args:
        array_data: Array data of shape (num_antennas, num_samples).

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for antenna_idx in range(array_data.shape[0]):
        ax.plot(array_data[antenna_idx], label=f"Antenna {antenna_idx}")

    ax.set_title("Baseband Signal (Real Projection) per Antenna")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_fft_spectrum(
    freqs_hz: np.ndarray,
    spectrum_db: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(freqs_hz / 1e6, spectrum_db)
    ax.set_title("FFT Spectrum")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_spectrogram_waterfall(
    freqs_hz: np.ndarray,
    times_s: np.ndarray,
    spec_db: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))

    mesh = ax.pcolormesh(
        times_s,
        freqs_hz / 1e6,
        spec_db,
        shading="auto",
        cmap="jet",   
    )

    ax.set_title("Spectrogram / Waterfall")
    ax.set_xlabel("Time (ms)")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([f"{t*1000:.2f}" for t in ax.get_xticks()])
    ax.set_ylabel("Frequency (MHz)")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Magnitude (dB)")

    fig.tight_layout()
    return fig


def plot_energy_gauge(
    energy_value: float,
    threshold_value: float,
) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(7, 1.8))

    # Normalize for visualization
    max_value = max(energy_value, threshold_value) * 1.5

    # Threshold line
    ax.axvline(threshold_value, color="black", linestyle="--", linewidth=2, label="Threshold")

    # Color based on detection
    if energy_value >= threshold_value:
        bar_color = "green"
        status_text = "DETECTED"
    else:
        bar_color = "red"
        status_text = "NO SIGNAL"

    ax.barh([0], [energy_value], height=0.4, color=bar_color, label="Energy")

    ax.set_xlim(0, max_value)
    ax.set_yticks([])
    ax.set_xlabel("Energy")

    ax.set_title(f"Energy Detection (Fixed Threshold) Status: {status_text}")

    ax.legend(loc="upper right")
    ax.grid(True, axis="x")

    fig.tight_layout()
    return fig

def plot_cfar_threshold_view(
    sample_indices: np.ndarray,
    instantaneous_power: np.ndarray,
    cfar_thresholds: np.ndarray,
    cfar_detections: np.ndarray,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.plot(sample_indices, instantaneous_power, label="Instantaneous Power")
    ax.plot(sample_indices, cfar_thresholds, linestyle="--", label="CFAR Threshold")

    ax.scatter(
        sample_indices[cfar_detections],
        instantaneous_power[cfar_detections],
        color='black',
        s=15,
        label="CFAR Detections",
    )

    ax.set_title("CA-CFAR Detection: Adaptive Threshold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Power |x[n]|²")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_radar_scope_view(
    true_angles_deg=None,
    estimated_angles_deg=None,
    sweep_angle_deg=30,
    title="RF Bearing Radar Scope",
    max_range=1.0,
):
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_ylim(0, max_range)

    ax.grid(True, color="lime", alpha=0.30)
    ax.tick_params(colors="lime", labelsize=8)
    ax.spines["polar"].set_color("lime")
    ax.set_yticklabels([])

    # Radar sweep sector
    sweep_width = 18
    sweep_start = sweep_angle_deg - sweep_width
    sweep_end = sweep_angle_deg

    theta = np.deg2rad(np.linspace(sweep_start, sweep_end, 80))
    r = np.linspace(0, max_range, 80)

    for rr in r:
        ax.plot(theta, np.full_like(theta, rr), color="lime", alpha=0.035)

    # Main sweep line
    sweep_theta = np.deg2rad(sweep_angle_deg)
    ax.plot(
        [sweep_theta, sweep_theta],
        [0, max_range],
        color="lime",
        linewidth=2.0,
        alpha=0.9,
    )

    # True bearing lines
    if true_angles_deg is not None:
        for angle in true_angles_deg:
            theta_angle = np.deg2rad(angle)
            ax.plot(
                [theta_angle, theta_angle],
                [0, max_range],
                linestyle="--",
                color="cyan",
                linewidth=1.2,
                alpha=0.8,
            )

    # Estimated emitter dots
    if estimated_angles_deg is not None:
        ranges = np.linspace(0.60, 0.88, len(estimated_angles_deg))

        for angle, rr in zip(estimated_angles_deg, ranges):
            theta_angle = np.deg2rad(angle)

            ax.scatter(theta_angle, rr, s=450, color="red", alpha=0.18)
            ax.scatter(theta_angle, rr, s=220, color="red", alpha=0.35)
            ax.scatter(theta_angle, rr, s=80, color="white", alpha=0.7)
            ax.scatter(theta_angle, rr, s=45, color="red", alpha=1.0)

    ax.set_title(title, color="white", pad=18, fontsize=12)

    return fig