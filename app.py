"""
----------------------------------------------------------------------
RFDetect Lab - Signal Processing Toolkit
----------------------------------------------------------------------

Author: Dr. Ing. Evans Baidoo
Organization: Personal Project 
Date Created: 2026-03-25

Description:
RFDetect Lab provides tools for RF signal detection, spectral analysis,
and angle-of-arrival estimation using simulated baseband IQ data.

----------------------------------------------------------------------
Copyright (c) 2026 Dr. Ing. Evans Baidoo
All Rights Reserved.
----------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from src.pipeline import run_analysis_pipeline, run_aoa_method_for_comparison

from config import ANGLE_GRID_DEG, NUM_ANTENNAS, ANTENNA_SPACING_M, WAVELENGTH_M

from src.plotting import (
    plot_spatial_spectrum,
    plot_channel_real_part,
    plot_fft_spectrum,
    plot_spectrogram_waterfall,
    plot_cfar_threshold_view,
    plot_energy_gauge,
    plot_radar_scope_view,
)

st.markdown("""
<style>

/* === GLOBAL LAYOUT === */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0.5rem !important;
}

/* Sidebar spacing */
section[data-testid="stSidebar"] > div {
    padding-top: 0.5rem !important;
}

/* === HEADER TIGHTENING === */
h1 {
    margin-top: 0px !important;
    margin-bottom: 0.3rem !important;
}

h2, h3, h4 {
    margin-top: 0.2rem !important;
    margin-bottom: 0.3rem !important;
}

/* Subtitle (the gray text under title) */
p {
    margin-top: 0px !important;
    margin-bottom: 0.5rem !important;
}

/* === METRICS === */
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
}

[data-testid="stMetricValue"] {
    font-size: 18px !important;
}

/* === GENERAL TEXT === */
body {
    font-size: 13px;
}

/* === ALERT BOXES (GREEN BARS) === */
div[data-testid="stAlert"] {
    padding-top: 0.4rem !important;
    padding-bottom: 0.4rem !important;
    margin-top: 0.3rem !important;
    margin-bottom: 0.3rem !important;
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="RF Bearing Demo", layout="wide")

st.title("RF Emitter Direction Finding Demo")
# st.title("RF Direction Finding Tool ")
st.caption("Interactive RF Direction-Finding Tool for Multi-Emitter Analysis ")
# st.caption("Synthetic multi-antenna system with detection, beamforming, and confidence metrics (Python, Streamlit)")

CUSTOM_DEFAULTS = {
    "preset_name": "Custom",
    "signal_type": "tone",
    "num_antennas": NUM_ANTENNAS,
    "num_emitters": 2,
    "estimation_method": "MUSIC",
    "num_sources": 2,
    "min_peak_separation_deg": 8,
    "peak_mode": "Threshold-Based",
    "max_reported_peaks": 1,
    "peak_threshold": 0.3,
    "smoothing": False,
    "snr_db": 20,
    "cfar_detection_scale": 1.0,
    "spectrum_window_type": "hann",
    "spectrogram_segment_length": 125,
    "spectrogram_overlap_length": 64,
    "num_samples": 256,
    "detection_threshold": 0.3,
    "detection_type": "Fixed threshold",
    "confidence_threshold": 2.0,
    "frame_length": 64,
    "angle_0": 25,
    "angle_1": -30,
    "angle_2": 40,
    "amp_0": 1.0,
    "amp_1": 0.8,
    "amp_2": 0.6,
    "freq_0": 100_000,
    "freq_1": 120_000,
    "freq_2": 140_000,
}


PRESET_SCENARIOS = {
    "Single emitter - clean tone": {
        "signal_type": "tone", "num_emitters": 1, "snr_db": 25,
        "num_samples": 256, "detection_type": "Fixed threshold",
        "detection_threshold": 0.3, "cfar_detection_scale": 1.0,
        "confidence_threshold": 2.0, "frame_length": 64,
        "estimation_method": "Beamforming", "peak_mode": "Threshold-Based",
        "max_reported_peaks": 1, "peak_threshold": 0.3,
        "min_peak_separation_deg": 8, "num_sources": 1, "smoothing": False,
        "angle_0": 25, "amp_0": 1.0, "freq_0": 100_000,
    },
    "Two emitters - separated bearings": {
        "signal_type": "tone", "num_emitters": 2, "snr_db": 20,
        "num_samples": 256, "detection_type": "Fixed threshold",
        "detection_threshold": 0.3, "cfar_detection_scale": 1.0,
        "confidence_threshold": 2.0, "frame_length": 64,
        "estimation_method": "Beamforming", "peak_mode": "Threshold-Based",
        "max_reported_peaks": 2, "peak_threshold": 0.3,
        "min_peak_separation_deg": 8, "num_sources": 2, "smoothing": False,
        "angle_0": 25, "angle_1": -8,
        "amp_0": 1.0, "amp_1": 0.7,
        "freq_0": 100_000, "freq_1": 109_325,
    },
    "Two emitters - close-angle challenge": {
        "signal_type": "tone", "num_emitters": 2, "snr_db": 20,
        "num_samples": 512, "detection_type": "Fixed threshold",
        "detection_threshold": 0.3, "cfar_detection_scale": 1.0,
        "confidence_threshold": 2.0, "frame_length": 128,
        "estimation_method": "MUSIC", "peak_mode": "Top-N Peaks",
        "max_reported_peaks": 2, "peak_threshold": 0.25,
        "min_peak_separation_deg": 3, "num_sources": 2, "smoothing": False,
        "angle_0": 10, "angle_1": 15,
        "amp_0": 1.0, "amp_1": 0.9,
        "freq_0": 100_000, "freq_1": 112_000,
    },
    "Three emitters - unequal power": {
        "signal_type": "tone", "num_emitters": 3, "snr_db": 20,
        "num_samples": 512, "detection_type": "Fixed threshold",
        "detection_threshold": 0.3, "cfar_detection_scale": 1.0,
        "confidence_threshold": 2.0, "frame_length": 128,
        "estimation_method": "MUSIC", "peak_mode": "Top-N Peaks",
        "max_reported_peaks": 3, "peak_threshold": 0.2,
        "min_peak_separation_deg": 6, "num_sources": 3, "smoothing": False,
        "angle_0": -35, "angle_1": 8, "angle_2": 42,
        "amp_0": 1.0, "amp_1": 0.65, "amp_2": 0.4,
        "freq_0": 95_000, "freq_1": 125_000, "freq_2": 155_000,
    },
    "Low SNR - CFAR stress test": {
        "signal_type": "tone", "num_emitters": 2, "snr_db": 0,
        "num_samples": 512, "detection_type": "CFAR",
        "detection_threshold": 0.3, "cfar_detection_scale": 2.0,
        "confidence_threshold": 2.0, "frame_length": 128,
        "estimation_method": "Beamforming", "peak_mode": "Threshold-Based",
        "max_reported_peaks": 2, "peak_threshold": 0.25,
        "min_peak_separation_deg": 8, "num_sources": 2, "smoothing": False,
        "angle_0": 25, "angle_1": -20,
        "amp_0": 1.0, "amp_1": 0.8,
        "freq_0": 100_000, "freq_1": 135_000,
    },
    "Noise only - false alarm check": {
        "signal_type": "noise_only", "num_emitters": 1, "snr_db": -5,
        "num_samples": 256, "detection_type": "CFAR",
        "detection_threshold": 0.3, "cfar_detection_scale": 4.0,
        "confidence_threshold": 2.0, "frame_length": 64,
        "estimation_method": "Beamforming", "peak_mode": "Threshold-Based",
        "max_reported_peaks": 1, "peak_threshold": 0.3,
        "min_peak_separation_deg": 8, "num_sources": 1, "smoothing": False,
        "angle_0": 25, "amp_0": 1.0, "freq_0": 100_000,
    },
}


def apply_selected_preset():
    preset_name = st.session_state.get("preset_name", "Custom")
    if preset_name == "Custom":
        return

    for key, value in PRESET_SCENARIOS[preset_name].items():
        st.session_state[key] = value


def reset_to_custom_defaults():
    for key, value in CUSTOM_DEFAULTS.items():
        st.session_state[key] = value


def initialize_defaults():
    for key, value in CUSTOM_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_defaults()

# Sidebar controls
st.sidebar.header("Scenario Settings")

st.sidebar.selectbox(
    "Scenario Preset",
    options=["Custom"] + list(PRESET_SCENARIOS.keys()),
    key="preset_name",
    on_change=apply_selected_preset,
    help="Load a ready-made RF scenario. Choose Custom for manual tuning.",
)

col_run, col_reset = st.sidebar.columns(2)

with col_run:
    run_button = st.button("Run Analysis", use_container_width=True)

with col_reset:
    st.button(
        "Reset Controls",
        use_container_width=True,
        on_click=reset_to_custom_defaults,
    )
    
signal_type = st.sidebar.selectbox(
    "Signal Type",
    options=["tone", "burst", "noise_only"],
    key="signal_type",
)

st.sidebar.markdown("### 📡 Signal Settings")

col1, col2 = st.sidebar.columns(2)
with col1:
    snr_db = st.slider("SNR (dB)", -10, 30, key="snr_db")

with col2:
    num_samples = st.slider("Samples", 128, 4096, step=128, key="num_samples")

with st.sidebar.expander("Array Configuration", expanded=False):
    num_antennas = st.selectbox(
        "Number of Antennas",
        options=[2, 4, 6, 8, 12],
        key="num_antennas",
        help="Number of ULA antenna elements used for signal generation and AoA estimation.",
    )


with st.sidebar.expander("Spectrum Analysis", expanded=False):
    spectrum_window_type = st.selectbox(
        "FFT / Spectrogram Window",
        options=["rectangular", "hann", "hamming", "blackman"],
        key="spectrum_window_type",
        help="Select the window used for FFT and spectrogram analysis.",
    )

    spectrogram_segment_length = st.selectbox(
        "Spectrogram Window Length",
        options=[128, 256, 512, 1024],
        key="spectrogram_segment_length",
        help="Number of samples used in each spectrogram FFT window.",
    )

    spectrogram_overlap_length = st.selectbox(
        "Spectrogram Overlap",
        options=[0, 64, 128, 192, 256, 384, 512, 768],
        key="spectrogram_overlap_length",
        help="Number of samples shared between adjacent spectrogram windows.",
    )

    if spectrogram_overlap_length >= spectrogram_segment_length:
        st.warning("Overlap must be smaller than the window length. It will be clipped automatically.")
        
st.sidebar.markdown("### Detection Settings")
detection_type = st.sidebar.selectbox(
    "Dection Method",
    options=["Fixed threshold", "CFAR"],
    key="detection_type",
)

col1, col2 = st.sidebar.columns(2)
with col1:
    detection_threshold = st.slider(
        "Threshold", 0.1, 1.0, step=0.1, key="detection_threshold"
    )

with col2:
    cfar_detection_scale = st.slider(
        "CFAR Detection Scale",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        key="cfar_detection_scale",
        help="Multiplier applied to the local noise estimate for sample-wise CFAR detection.",
    )
    
col3, col4 = st.sidebar.columns(2)
with col3:
    frame_length = st.select_slider(
        "Frame Length",
        options=[32, 64, 128, 256],
        key="frame_length",
    )
with col4:
    confidence_threshold = st.slider(
        "Confidence", 0.0, 20.0, step=0.1, key="confidence_threshold"
    )

num_emitters = st.sidebar.selectbox(
    "Number of Emitters",
    options=[1, 2, 3],
    index=0,
    key="num_emitters",
)


emitter_angles_deg = []
emitter_amplitudes = []
emitter_frequencies_hz = []


st.sidebar.markdown("### Emitter Parameters")

for emitter_idx in range(num_emitters):
    with st.sidebar.expander(f"Emitter {emitter_idx + 1}", expanded=(emitter_idx == 0)):

        col1, col2, col3 = st.columns(3)

        with col1:
            angle_deg = st.slider(
                "Angle (deg)",
                -90, 90,
                key=f"angle_{emitter_idx}"
            )

        with col2:
            amplitude = st.slider(
                "Amplitude",
                0.1, 2.0,
                key=f"amp_{emitter_idx}"
            )

        with col3:
            frequency_hz = st.slider(
                "Frequency",
                1000, 200000,
                key=f"freq_{emitter_idx}"
            )

    emitter_angles_deg.append(float(angle_deg))
    emitter_amplitudes.append(float(amplitude))
    emitter_frequencies_hz.append(float(frequency_hz))

estimation_method = st.sidebar.selectbox(
    "Estimation Method",
    options=["Beamforming", "MUSIC", "Phase-Difference"],
    key="estimation_method",
)


num_sources = 1
if estimation_method == "MUSIC":
    num_sources = st.sidebar.selectbox(
        "Number of Sources",
        options=[1, 2, 3],
        key="num_sources",
        help="Required for MUSIC to define the signal subspace dimension.",
    )

elif estimation_method == "Beamforming":
    min_peak_separation_deg = st.sidebar.slider(
        "Minimum Peak Separation (deg)",
        min_value=1,
        max_value=20,
        step=1,
        key="min_peak_separation_deg",
        help="Controls how close two detected peaks can be.",
    )

st.sidebar.markdown("### Peak Selection")
peak_mode = st.sidebar.selectbox(
    "Mode",
    options=["Top-N Peaks", "Threshold-Based"],
    key="peak_mode",
)

max_reported_peaks = 1
peak_threshold = 0.3
if peak_mode == "Top-N Peaks":
    max_reported_peaks = st.sidebar.selectbox(
        "Number of Peaks",
        options=[1, 2, 3, 4],
        key="max_reported_peaks",
    )
elif peak_mode == "Threshold-Based":
    peak_threshold = st.sidebar.slider(
        "Peak Threshold",
        min_value=0.1,
        max_value=1.0,
        step=0.05,
        key="peak_threshold",
    )

smoothing = st.sidebar.toggle("Spatial Smoothing", value=False, key="smoothing")


spectrogram_frequency_axis_hz = None
spectrogram_time_axis_s = None
spectrogram_db = None
fft_frequency_axis_hz = None
fft_spectrum_db = None
spectrum_metrics = None
mean_angle_error = None
min_peak_separation_deg = st.session_state.get("min_peak_separation_deg", 8)


def plot_framewise_bearing_detections(frame_indices, frame_detected_angles_deg, true_angles=None):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    
    for frame_idx, angles in zip(frame_indices, frame_detected_angles_deg):
        if len(angles) > 0:
            ax.scatter(
                np.full(len(angles), frame_idx),
                angles,
                color="red",
                marker="o",
                label="Detected Bearings" if frame_idx == frame_indices[0] else None,
            )

    if true_angles is not None:
        for idx, angle in enumerate(true_angles):
            ax.axhline(
                angle,
                linestyle="--",
                label="True Angle" if idx == 0 else None,
            )

    ax.set_title("Frame-wise Bearing Detections")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Angle (degrees)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def render_animated_radar_scope(
    true_angles_deg=None,
    estimated_angles_deg=None,
    size_px=430,
):
    """
    Render an animated radar-scope view using HTML/CSS/SVG.

    Bearing-only visualization:
    - angle is meaningful
    - range is illustrative
    """
    center = size_px / 2
    max_radius = size_px * 0.43

    def angle_to_xy(angle_deg, radius):
        theta = np.deg2rad(float(angle_deg))
        x = center + radius * np.sin(theta)
        y = center - radius * np.cos(theta)
        return x, y

    true_svg = ""
    if true_angles_deg is not None:
        for angle in true_angles_deg:
            x, y = angle_to_xy(angle, max_radius * 0.95)
            true_svg += f"""
            <line x1="{center}" y1="{center}" x2="{x}" y2="{y}"
                  stroke="#00e5ff" stroke-width="2"
                  stroke-dasharray="6 5" opacity="0.85"/>
            """

    estimated_svg = ""
    if estimated_angles_deg is not None:
        ranges = np.linspace(0.55, 0.88, max(len(estimated_angles_deg), 1))

        for angle, range_scale in zip(estimated_angles_deg, ranges):
            x, y = angle_to_xy(angle, max_radius * range_scale)
            estimated_svg += f"""
            <circle cx="{x}" cy="{y}" r="18" fill="rgba(255,0,70,0.18)"/>
            <circle cx="{x}" cy="{y}" r="10" fill="rgba(255,0,70,0.45)"/>
            <circle cx="{x}" cy="{y}" r="5" fill="#ff3355"/>
            <circle cx="{x}" cy="{y}" r="2" fill="white"/>
            """

    html = f"""
    <div class="radar-wrapper">
        <div class="radar-scope">
            <div class="radar-sweep"></div>

            <svg width="{size_px}" height="{size_px}" viewBox="0 0 {size_px} {size_px}">
                <circle cx="{center}" cy="{center}" r="{max_radius * 0.25}" class="ring"/>
                <circle cx="{center}" cy="{center}" r="{max_radius * 0.50}" class="ring"/>
                <circle cx="{center}" cy="{center}" r="{max_radius * 0.75}" class="ring"/>
                <circle cx="{center}" cy="{center}" r="{max_radius}" class="outer-ring"/>

                <line x1="{center}" y1="{center - max_radius}" x2="{center}" y2="{center + max_radius}" class="grid-line"/>
                <line x1="{center - max_radius}" y1="{center}" x2="{center + max_radius}" y2="{center}" class="grid-line"/>

                {true_svg}
                {estimated_svg}
            </svg>
        </div>
        
    </div>

    <style>
    .radar-wrapper {{
        width: {size_px}px;
        margin: 0 auto;
        text-align: center;
        font-family: Arial, sans-serif;
    }}

    .radar-scope {{
        position: relative;
        width: {size_px}px;
        height: {size_px}px;
        border-radius: 50%;
        overflow: hidden;
        background:
            repeating-conic-gradient(
                rgba(0,255,80,0.16) 0deg 1deg,
                transparent 1deg 10deg
            ),
            radial-gradient(circle, rgba(0,255,80,0.16), rgba(0,20,5,0.95) 62%, #000 100%);
        box-shadow:
            0 0 25px rgba(0,255,80,0.35),
            inset 0 0 35px rgba(0,255,80,0.22);
        border: 2px solid #00ff55;
    }}

    .radar-sweep {{
        position: absolute;
        width: 50%;
        height: 50%;
        left: 50%;
        top: 50%;
        transform-origin: 0% 0%;
        background: linear-gradient(
            45deg,
            rgba(0,255,80,0.65),
            rgba(0,255,80,0.25) 35%,
            transparent 75%
        );
        clip-path: polygon(0 0, 100% 0, 100% 100%);
        animation: sweepRotate 2.2s linear infinite;
        z-index: 1;
    }}

    svg {{
        position: absolute;
        top: 0;
        left: 0;
        z-index: 2;
    }}

    .ring {{
        fill: none;
        stroke: rgba(0,255,80,0.30);
        stroke-width: 1.2;
    }}

    .outer-ring {{
        fill: none;
        stroke: rgba(0,255,80,0.85);
        stroke-width: 2;
    }}

    .grid-line {{
        stroke: rgba(0,255,80,0.22);
        stroke-width: 1;
    }}

    .radar-caption {{
        margin-top: 8px;
        color: #9affb6;
        font-size: 12px;
        opacity: 0.85;
    }}

    @keyframes sweepRotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    </style>
    """

    components.html(html, height=size_px + 45)


if run_button:
    pipeline_config = {
        "signal_type": signal_type,
        "num_emitters": num_emitters,
        "emitter_angles_deg": emitter_angles_deg,
        "emitter_amplitudes": emitter_amplitudes,
        "emitter_frequencies_hz": emitter_frequencies_hz,
        "snr_db": snr_db,
        "num_samples": num_samples,
        "num_antennas": num_antennas,
        "detection_type": detection_type,
        "detection_threshold": detection_threshold,
        "cfar_detection_scale": cfar_detection_scale,
        "confidence_threshold": confidence_threshold,
        "estimation_method": estimation_method,
        "num_sources": num_sources,
        "peak_mode": peak_mode,
        "min_peak_separation_deg": min_peak_separation_deg,
        "max_reported_peaks": max_reported_peaks,
        "peak_threshold": peak_threshold,
        "frame_length": frame_length,
        "smoothing": smoothing,
        "spectrum_window_type": spectrum_window_type,
        "spectrogram_segment_length": spectrogram_segment_length,
        "spectrogram_overlap_length": spectrogram_overlap_length,
    }

    results = run_analysis_pipeline(pipeline_config)
    
    array_data = results["array_data"]
    processed_array_data = results["processed_array_data"]

    detected = results["detected"]
    energy = results["energy"]
    instantaneous_power = results["instantaneous_power"]
    sample_cfar_detections = results["sample_cfar_detections"]
    sample_cfar_thresholds = results["sample_cfar_thresholds"]
    sample_indices = results["sample_indices"]

    fft_freqs_hz = results["fft_freqs_hz"]
    fft_spectrum_db = results["fft_spectrum_db"]
    spectrogram_frequency_axis_hz = results["spectrogram_frequency_axis_hz"]
    spectrogram_time_axis_s = results["spectrogram_time_axis_s"]
    spectrogram_db = results["spectrogram_db"]
    spectrum_metrics = results["spectrum_metrics"]

    spatial_spectrum = results["spatial_spectrum"]
    detected_angles_deg = results["detected_angles_deg"]
    detected_peak_values = results["detected_peak_values"]
    mean_angle_error = results["mean_angle_error"]
    confidence_score = results["confidence_score"]
    par = results["par"]
    pslr = results["pslr"]
    phase_difference_rad = results["phase_difference_rad"]
    peak_power = results["peak_power"]

    frame_indices = results["frame_indices"]
    frame_detected_angles_deg = results["frame_detected_angles_deg"]
    frame_detected_peak_values = results["frame_detected_peak_values"]

    true_angles_for_display = results["true_angles_for_display"]
    resolution_diagnostics = results["resolution_diagnostics"]
    coherent_source_diagnostics = results["coherent_source_diagnostics"]
    is_reliable = results["is_reliable"]


    if is_reliable:
        st.success("Reliable emitter detected.")
    elif detected:
        st.warning("Emitter detected, but estimate is unreliable.")
    else:
        st.error("No reliable emitter detected.")

    if estimation_method == "Phase-Difference" and num_emitters > 1:
        st.warning("⚠️ Phase-Difference AoA unstable — possible multiple emitters.")
   
    with st.expander("Active Scenario Configuration", expanded=False):
        true_angle_text = ", ".join(f"{angle:.1f}°" for angle in emitter_angles_deg)
        st.markdown(f"""
        - **Preset:** `{st.session_state.get("preset_name", "Custom")}`
        - **Signal type:** `{signal_type}`
        - **Emitters:** `{num_emitters}`
        - **True angles:** `{true_angle_text}`
        - **Detection method:** `{detection_type}`
        - **AoA method:** `{estimation_method}`
        - **SNR:** `{snr_db} dB`
        - **Antennas:** `{num_antennas}`
        """)
   
    tab_overview, tab_spectrum, tab_detection, tab_comparison, tab_metrics = st.tabs(
    ["Overview", "Spectrum", "Detection", "Method Comparison", "Metrics"]
    )
    with tab_overview:
        st.subheader("AoA Overview")
        
        # col_left, col_right = st.columns(2)

        # with col_left:
        st.markdown("#### RF Bearing Radar Scope")

        if spatial_spectrum is not None:
            render_animated_radar_scope(
                true_angles_deg=true_angles_for_display,
                estimated_angles_deg=detected_angles_deg,
                size_px=350,
            )
        else:
            st.warning("No signal detected")

                
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### Received Baseband signal")
            st.pyplot(plot_channel_real_part(processed_array_data), use_container_width=True)
              
        with col_right:
            if estimation_method == "Phase-Difference":
                st.info("Frame-wise multi-detection view is not enabled for Phase-Difference AoA.")
            else:
                st.markdown("#### Frame-wise Bearing Detections")
                if len(frame_indices) > 0:
                    st.pyplot(
                        plot_framewise_bearing_detections(
                            frame_indices,
                            frame_detected_angles_deg,
                            true_angles=true_angles_for_display,
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("No frame-wise detections available.")

    with tab_spectrum:
        st.subheader("FFT Spectrum and Spectrogram")

        col1, col2, col3 = st.columns(3)
        if spectrum_metrics is not None:
            col1.metric("Peak Frequency", f"{spectrum_metrics['peak_frequency_hz']/1e6:.3f} MHz")
            col2.metric("Peak Power", f"{spectrum_metrics['peak_power_db']:.2f} dB")
            col3.metric("Mean Level", f"{spectrum_metrics['mean_level_db']:.2f} dB")
        else:
            col1.metric("Peak Frequency", "N/A")
            col2.metric("Peak Power", "N/A")
            col3.metric("Mean Level", "N/A")

        st.markdown("#### FFT Spectrum")  
        if not detected:
            st.warning("⚠️ No signal detected — spectrum shows noise only")

        st.pyplot(
            plot_fft_spectrum(fft_freqs_hz, fft_spectrum_db),
            use_container_width=True,
        )    
            
        st.pyplot(plot_spectrogram_waterfall(
            spectrogram_frequency_axis_hz,
            spectrogram_time_axis_s,
            spectrogram_db
        ),
            use_container_width=True,)        

    with tab_detection:
        st.subheader("Detection View")

        st.markdown("#### Energy Detection")
        st.pyplot(
            plot_energy_gauge(
                energy_value=energy,
                threshold_value=detection_threshold,
            ),
            use_container_width=True,
        )
        
        st.markdown("#### CA-CFAR Adaptive Detection")
        st.pyplot(
            plot_cfar_threshold_view(
                sample_indices=sample_indices,
                instantaneous_power=instantaneous_power,
                cfar_thresholds=sample_cfar_thresholds,
                cfar_detections=sample_cfar_detections,
            ),
            use_container_width=True,
        )
        

    with tab_comparison:
        st.subheader("AoA Method Comparison")

        if not detected:
            st.warning("No signal detected. Method comparison is skipped because there is no reliable signal to estimate.")
        else:
            comparison_methods = ["Beamforming", "MUSIC", "Phase-Difference"]
            comparison_max_peaks = max(1, min(int(num_emitters), int(max_reported_peaks)))
            comparison_source_setting = st.session_state.get("num_sources", num_emitters)
            comparison_num_sources = max(1, min(int(comparison_source_setting), int(num_emitters), num_antennas - 1))

            comparison_results = [
                run_aoa_method_for_comparison(
                    method_name=method_name,
                    processed_array_data=processed_array_data,
                    true_angles_for_display=true_angles_for_display,
                    num_sources=comparison_num_sources,
                    peak_mode=peak_mode,
                    min_peak_separation_deg=min_peak_separation_deg,
                    max_reported_peaks=comparison_max_peaks,
                    peak_threshold=peak_threshold,
                    smoothing=smoothing,
                    num_antennas=num_antennas,
                )
                for method_name in comparison_methods
            ]

            table_rows = []
            for result in comparison_results:
                angle_text = (
                    ", ".join(f"{angle:.2f}°" for angle in result["detected_angles_deg"])
                    if len(result["detected_angles_deg"]) > 0
                    else "N/A"
                )
                table_rows.append(
                    {
                        "Method": result["method"],
                        "Estimated Angle(s)": angle_text,
                        "Mean Error (deg)": f"{result['mean_error_deg']:.2f}" if result["mean_error_deg"] is not None else "N/A",
                        "Runtime (ms)": f"{result['runtime_ms']:.2f}",
                        "Confidence (%)": f"{result['confidence_score']:.2f}" if result["confidence_score"] is not None else "N/A",
                        "Status": result["status"],
                        "Notes": result["notes"],
                    }
                )
            emitter_pos = ", ".join(f"{x:.1f}°" for x in true_angles_for_display) if true_angles_for_display is not None else "N/A" 
            
            col1, col2,= st.columns(2)
            col1.metric("Emitter True angle(s)", emitter_pos)

            st.table(table_rows)

            st.markdown("#### Spatial Spectrum Comparison")
            for result in comparison_results:
                with st.expander(result["method"], expanded=(result["method"] == estimation_method)):
                    if result["spatial_spectrum"] is None:
                        st.warning(result["notes"])
                    else:
                        st.pyplot(
                            plot_spatial_spectrum(
                                angle_grid_deg=ANGLE_GRID_DEG,
                                spatial_spectrum=result["spatial_spectrum"],
                                true_angles_deg=true_angles_for_display,
                                estimated_angles_deg=result["detected_angles_deg"],
                                estimation_method=result["method"],
                            ),
                            use_container_width=True,
                        )

            st.info(
                "The comparison uses the same generated array data, detection result, peak-selection settings, "
                "and scenario configuration. Beamforming and MUSIC are spectrum-based methods; "
                "Phase-Difference is fastest but is mainly valid for a single dominant emitter."
            )

    with tab_metrics:
        st.subheader("Metrics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Detection", "Yes" if detected else "No")               
        col2.metric("Confidence (%)", f"{confidence_score:.2f}" if confidence_score is not None else "N/A")
        col3.metric("Energy", f"{energy:.4f}")
        col4.metric(
                    "Coherence Status",
                    coherent_source_diagnostics["status"],
                )

        col5, col6, col7, col8 = st.columns(4)
        col5.metric(
                    "Minimum Emitter Separation",
                    (
                        f"{resolution_diagnostics['min_separation_deg']:.2f}°"
                        if resolution_diagnostics["min_separation_deg"] is not None
                        else "N/A"
                    ),
                )

        col6.metric(
            "Approx. Beamforming Beamwidth",
            (
                f"{resolution_diagnostics['beamwidth_deg']:.2f}°"
                if resolution_diagnostics["beamwidth_deg"] is not None
                else "N/A"
            ),
        )

        col7.metric(
            "Minimum Frequency Spacing",
            (
                f"{coherent_source_diagnostics['min_frequency_spacing_hz']:.0f} Hz"
                if coherent_source_diagnostics["min_frequency_spacing_hz"] is not None
                else "N/A"
            ),
        )

        col8.metric(
                    "Resolution Status",
                    resolution_diagnostics["status"],
                )
       
        with st.expander("Emitter Resolution Diagnostics", expanded=True):
            if not resolution_diagnostics["enabled"]:
                st.info(resolution_diagnostics["message"])
            if resolution_diagnostics["status"] == "Warning":
                st.warning(resolution_diagnostics["message"])
            else:
                st.success(resolution_diagnostics["message"])

            if coherent_source_diagnostics["status"] == "Warning":
                st.warning(coherent_source_diagnostics["message"])
            elif coherent_source_diagnostics["status"] == "Caution":
                st.warning(coherent_source_diagnostics["message"])
            else:
                st.success(coherent_source_diagnostics["message"])

        # with st.expander("Coherent Source Warning", expanded=True):
        #     if not coherent_source_diagnostics["enabled"]:
        #         st.info(coherent_source_diagnostics["message"])
        #     else:
        #         col1, col2 = st.columns(2)

        #         col1.metric(
        #             "Minimum Frequency Spacing",
        #             f"{coherent_source_diagnostics['min_frequency_spacing_hz']:.0f} Hz",
        #         )

        #         col2.metric(
        #             "Coherence Status",
        #             coherent_source_diagnostics["status"],
        #         )

        #         if coherent_source_diagnostics["status"] == "Warning":
        #             st.warning(coherent_source_diagnostics["message"])
        #         elif coherent_source_diagnostics["status"] == "Caution":
        #             st.warning(coherent_source_diagnostics["message"])
        #         else:
        #             st.success(coherent_source_diagnostics["message"])
   
with st.expander("Project Summary (Assumptions)"):
    st.markdown("""
    The project follows the pipeline as follows:
                
        multi-channel IQ input → signal detection → direction estimate → visual output
    - The system operates on baseband IQ samples after RF downconversion and sampling.
    - A narrowband signal model is assumed (single-tone / narrowband approximation), enabling phase-based AoA estimation.
    - The antenna array is a Uniform Linear Array (ULA) with known geometry and ideal calibration (no gain/phase mismatch).
    - Supports single-emitter and multi-emitter synthetic scenarios through a configurable UI.
    - Three AoA estimation methods are implemented:
        - Phase-Difference AoA (single dominant source, fast, low complexity)
        - Beamforming (DBF) (scan-based baseline estimator)
        - MUSIC (high-resolution subspace method requiring known source count)
    - Phase-Difference AoA includes a consistency check across antenna pairs:
        - Consistent phase estimates → likely single dominant emitter
        - Inconsistent phase estimates → possible multiple emitters or corrupted data
    - Multi-target detection is supported via spatial spectrum peak extraction:
        - Local peak selection or threshold-based peak detection
        - Minimum peak separation enforces physical target distinctness
    - Frame-wise processing provides temporal insight:
        - Outputs multiple bearing detections per frame
        - Current implementation does not perform track association (no persistent target IDs)
    - The propagation environment is assumed ideal:
        - No multipath, reflections, or interference
        - No mutual coupling between antennas
    - Noise is modeled as additive white Gaussian noise (AWGN) with configurable SNR.
    - Detection is based on a fixed-threshold energy detector (non-adaptive):
        - May produce false alarms under high noise if not tuned properly
    - Beamforming and MUSIC assume:
        - perfect synchronization
        - known wavelength
        - stationary signal within each frame
    - The scan range is limited to azimuth-only estimation (-90° to +90°)
    - The system uses synthetic data only and is not yet validated with real RF measurements.
    """)
