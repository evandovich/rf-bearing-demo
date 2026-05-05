"""
Global configuration for the RF Bearing Estimation Demo
"""

import numpy as np

# =========================================================
# Physical constants
# =========================================================
C = 299_792_458.0  # Speed of light (m/s)

# =========================================================
# RF / Array parameters
# =========================================================
NUM_ANTENNAS = 4

CARRIER_FREQUENCY_HZ = 2.4e9  # 2.4 GHz
WAVELENGTH_M = C / CARRIER_FREQUENCY_HZ

ANTENNA_SPACING_M = WAVELENGTH_M / 2  # lambda / 2 spacing

# =========================================================
# Signal parameters
# =========================================================
SAMPLE_RATE_HZ = 1e6  # 1 MHz
FRAME_LENGTH = 1024   # number of samples per frame

DEFAULT_SIGNAL_FREQUENCY_HZ = 100e3  # 100 kHz tone
DEFAULT_SNR_DB = 10.0

# =========================================================
# Angle grid (for beamforming scan)
# =========================================================
ANGLE_GRID_DEG = np.arange(-90, 91, 1)

# =========================================================
# Signal types
# =========================================================
SIGNAL_TYPE_TONE = "tone"
SIGNAL_TYPE_BURST = "burst"
SIGNAL_TYPE_NOISE_ONLY = "noise_only"

# =========================================================
# Default scenario
# =========================================================
DEFAULT_TRUE_ANGLE_DEG = 20.0
DEFAULT_DETECTION_THRESHOLD = 0.1

# =========================================================
# Reproducibility
# =========================================================
RANDOM_SEED = 42