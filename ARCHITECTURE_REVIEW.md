# Architecture Notes

## Overall impression

The project is a strong demo foundation. It already separates the major DSP blocks into modules: signal generation, preprocessing, detection, AoA estimation, spectrum analysis, metrics, and plotting. The Streamlit interface makes the workflow easy to demonstrate, especially for explaining how SNR, emitter count, and estimator choice affect bearing quality.

## What is already good

- Clear DSP pipeline from synthetic IQ generation to visual output.
- Multi-emitter simulation is already supported.
- Multiple AoA methods are implemented: conventional beamforming, MUSIC, and phase-difference.
- Detection now includes both fixed-threshold and CA-CFAR style logic.
- The project has useful visualization layers: spatial spectrum, compass, FFT, spectrogram, energy gauge, and frame-wise detections.
- Functions are mostly grouped by purpose instead of being placed only in `app.py`.

## System Architecture

The RF Bearing Demo follows a modular signal-processing pipeline, from synthetic RF signal generation to direction-of-arrival estimation and visualization.

## Data Flow Summary
Synthetic IQ → Detection → AoA → Peaks → Metrics → UI

### High-Level Pipeline

```text
Signal Generation → Preprocessing → Detection → AoA Estimation → Post-processing → Visualization

+----------------------+
| Signal Generator     |
|----------------------|
| - Tone / Burst       |
| - Multi-emitter      |
| - SNR control        |
+----------+-----------+
           |
           v
+----------------------+
| Preprocessing        |
|----------------------|
| - DC removal         |
| - Normalization      |
| - Optional smoothing |
+----------+-----------+
           |
           v
+----------------------+
| Detection            |
|----------------------|
| - Energy detection   |
| - CA-CFAR            |
| - Frame segmentation |
+----------+-----------+
           |
           v
+----------------------+
| AoA Estimation       |
|----------------------|
| - Beamforming (DBF)  |
| - MUSIC              |
| - Phase-Difference   |
+----------+-----------+
           |
           v
+----------------------+
| Peak Extraction      |
|----------------------|
| - Top-N Peaks        |
| - Threshold-Based    |
| - Min separation     |
+----------+-----------+
           |
           v
+----------------------+
| Metrics & Diagnostics|
|----------------------|
| - Confidence score   |
| - PAR / PSLR         |
| - Resolution check   |
| - Coherence warning  |
+----------+-----------+
           |
           v
+----------------------+
| Visualization Layer  |
|----------------------|
| - Radar scope view   |
| - Spatial spectrum   |
| - Bearing compass    |
| - Frame-wise plots   |
| - FFT & spectrogram  |
+----------------------+.


