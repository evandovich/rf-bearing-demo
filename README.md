## Author

Created by Evans Baidoo.

Copyright © 2026 Evans Baidoo.

# RF Bearing Estimation Demo

A Streamlit-based RF direction-finding demo for simulating narrowband emitters received by a  variable element Uniform Linear Array (ULA), detecting signal presence, and estimating angle of arrival (AoA) using multiple DSP methods.

The project is designed as demo-ready lab: it shows the full chain from synthetic IQ generation to detection, spatial-spectrum estimation, multi-peak extraction, frame-wise bearing analysis, and visual interpretation.

---

## What This Project Does
## 🎬 Demo

<p align="center">
  <img src="rf_bearing_demo.gif.gif" width="800">
</p>

The demo follows this processing chain:

```text
Synthetic emitter setup
        ↓
Multi-channel IQ generation for a 4-element ULA
        ↓
Preprocessing: DC removal + normalization
        ↓
Signal detection: fixed energy threshold or sample-wise CFAR
        ↓
AoA estimation: Beamforming / MUSIC / Phase-Difference
        ↓
Peak extraction + frame-wise bearing view
        ↓
Streamlit visualization and reliability metrics
```

---

## Current Capabilities

### Scenario simulation

- Supports **1 to 3 synthetic RF emitters**
- Configurable emitter angle, amplitude, and tone frequency
- Supports tone, burst, and noise-only signal modes
- Uses variable-element ULA with half-wavelength antenna spacing
- Models additive white Gaussian noise (AWGN) with configurable SNR

### AoA estimation methods

The current implementation supports:

1. **Conventional Beamforming / DBF**
   - Scans an angle grid from -90° to +90°
   - Produces a spatial power spectrum
   - Works as the baseline AoA estimator

2. **MUSIC**
   - High-resolution subspace-based AoA estimator
   - Requires the assumed number of sources
   - Uses the sample covariance matrix and noise-subspace projection

3. **Phase-Difference AoA**
   - Fast low-complexity estimator
   - Uses average adjacent-antenna phase difference
   - Best suited for one dominant narrowband emitter
   - Includes per-antenna-pair angle estimates as a consistency check

### Detection methods

- Fixed-threshold energy detection
- Sample-wise CA-CFAR detection
- Detection tab displays instantaneous power and detection threshold behavior

### Multi-Emitter Support
- Simultaneous estimation of multiple emitters
- Peak-based detection:
  - Top-N Peaks
  - Threshold-Based
- Minimum peak separation control

### Visualization

#### Radar Scope View 
- Animated radar-style display
- Bearing-only visualization
- True angles (cyan dashed lines)
- Estimated emitters (red glowing points)

#### Spatial Spectrum
- Classic engineering view of AoA spectrum
- Available per method in comparison tab

#### Frame-wise Detection
- Temporal evolution of detected bearings

---

### Spectrum Analysis
- FFT spectrum
- Spectrogram (waterfall view)
- Configurable window types and overlap

---

### Diagnostics & Metrics

#### Core Metrics
- Detection status
- Confidence score
- Energy level
- Peak statistics

#### Resolution Diagnostics 
- Minimum emitter separation
- Approximate beamforming resolution limit
- Warning when emitters are too close

#### Coherent Source Warning 
- Detects when emitters share similar frequencies
- Flags potential MUSIC instability

---
### Scenario Presets
Pre-configured scenarios for testing:

- Single emitter (clean)
- Two emitters (separated)
- Two emitters (close-angle challenge)
- Three emitters (unequal power)
- Low SNR CFAR stress test
- Noise-only false alarm test

---

## Fixed Modeling Assumptions

This is a controlled synthetic demo, not a calibrated field receiver. The current assumptions are:
- Uniform Linear Array (ULA)
- Antenna spacing: λ / 2
- Signal model: narrowband complex baseband
- Emitters: 1 to 3 synthetic emitters
- Propagation: ideal far-field plane waves
- Noise: complex AWGN
- Synchronization: assumed perfect
- Input data: synthetic IQ only

---

## Signal Model

For one emitter, the received array signal is modeled as:

```text
x[n] = a(theta) * s[n] + w[n]
```

where:

- `s[n]` is the complex baseband source signal
- `a(theta)` is the ULA steering vector
- `w[n]` is additive complex Gaussian noise
- `x[n]` is the received multi-channel IQ data

For multiple emitters, the received signal is modeled as the sum of emitter contributions:

```text
x[n] = Σ a(theta_k) * A_k * s_k[n] + w[n]
```

where each emitter `k` has its own angle, amplitude, and source signal.

---

## ULA Steering Vector

For a 4-element ULA For example:

```text
a(theta) = [
    1,
    exp(-j * 2π * d * sin(theta) / λ),
    exp(-j * 2π * 2d * sin(theta) / λ),
    exp(-j * 2π * 3d * sin(theta) / λ)
]
```

where:

- `d` is the antenna spacing
- `λ` is the wavelength
- `theta` is the angle of arrival

---

## Project Structure

```text
rf_bearing_demo/
│── app.py                  # UI application
├── pipeline.py             # Main Application
├── config.py               # Global configuration
│
├── src/
│   ├── signal_generator.py # Signal + array simulation
│   ├── preprocessing.py    # DC removal, normalization, smoothing
│   ├── detection.py        # Energy + CFAR detection
│   ├── beamforming.py      # Beamforming, MUSIC, AoA logic
│   ├── spectrum_analysis.py# FFT and spectrogram
│   ├── metrics.py          # Performance metrics
│   ├── plotting.py         # Visualization utilities
│
└── requirements.txt
```

### Scenario Presets
Pre-configured scenarios for testing:

- Single emitter (clean)
- Two emitters (separated)
- Two emitters (close-angle challenge)
- Three emitters (unequal power)
- Low SNR CFAR stress test
- Noise-only false alarm test

---

Choose **Custom** when you want to manually tune the emitter count, angles, amplitudes, frequencies, detection settings, and AoA method.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

---

## Running the App

From the project folder:

```bash
streamlit run app.py
```

Then use the sidebar to configure:

- signal type
- SNR
- number of samples
- number of emitters
- each emitter angle, amplitude, and frequency
- detection method
- AoA estimation method
- peak selection method
- frame length and confidence threshold

---

## Current Limitations

- Synthetic data only
- No real SDR input yet
- No persistent target tracking or track IDs
- Phase-difference method is not reliable for multiple simultaneous emitters
- MUSIC requires the assumed number of sources
- Range is not estimated (bearing-only system)

---

## Roadmap

Possible next improvements:

- Add real IQ file input
- Multi-receiver triangulation (full localization)
- Real RF dataset integration
- Tracking (Kalman filter / multi-target tracking)
- GPU / C++ acceleration
- Sensor fusion (Radar + RF + Vision)
- Add pybind11 bridge for Python UI + C++ backend experimentation

---

## Dependencies

Main dependencies:

- Python
- NumPy
- SciPy
- Matplotlib
- Streamlit

See `requirements.txt` for the exact list.

---


