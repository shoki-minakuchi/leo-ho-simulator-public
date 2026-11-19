"""
System constants and simulation parameters for the LEO satellite network simulator.
"""

import math
import numpy as np

# =============================================================================
# 1. Simulation Control Parameters
# =============================================================================
SCALE_FACTORS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
MAX_DURATION = 1500.0
WARMUP_TIME = 20.0

# =============================================================================
# 2. Earth, Orbit, and Constellation Geometry Constants
# =============================================================================
R_E = 6371e3            # Earth radius [m]
H_SAT = 550e3           # Satellite orbital altitude [m]
R_ORB = R_E + H_SAT     # Orbital radius [m]
INCL_DEG = 53.0         # Orbital inclination [deg]
NUM_PLANES = 72         # Number of orbital planes
SATS_PER_PLANE = 22     # Satellites per plane
MEAN_MOTION = 2.0 * math.pi / (90.0 * 60.0)  # Mean motion [rad/s] (90-min period)
COS_INCL = math.cos(math.radians(INCL_DEG))
SIN_INCL = math.sin(math.radians(INCL_DEG))
EARTH_OMEGA = 7.2921159e-5  # Earth angular velocity [rad/s]

# =============================================================================
# 3. Handover & Protocol Constraints
# =============================================================================
MIN_ELEV = 25.0         # Minimum operational elevation angle [deg]
SAFE_ELEV = 30.0        # Conservative elevation threshold [deg]
HOF_PENALTY = 1.0       # Handover failure penalty duration [s]
RLF_PENALTY = 3.0       # Radio link failure penalty duration [s]
HOF_BACKOFF = 10.0      # Backoff interval after handover failure [s]

# Broadcast load-reporting (SIB) & hysteresis timer parameters
SIB_INTERVAL = 0.5      # Satellite load broadcast period [s]
RACH_DELAY_MAX = 0.05   # Maximum random access delay [s]
TIME_TO_STAY = 2.0      # Minimum stay duration after successful handover [s]

# =============================================================================
# 4. RF Link & Wireless Channel Parameters
# =============================================================================
P_TX_DBM = 65.0           # Transmit power [dBm]
G_TX_DBI = 35.0           # Transmit antenna gain [dBi]
G_RX_DBI = 32.0           # Receive antenna gain [dBi]
N_0_DBM = -100.0          # Thermal noise power [dBm]
PEN_LOSS_DB = 15.0        # Penetration / atmospheric loss [dB]
FC_MHZ = 20000.0          # Carrier frequency [MHz] (20 GHz Ka-band)
TOTAL_PRB = 2750          # Total Physical Resource Blocks per satellite
PRB_BW_MHZ = 0.36         # Bandwidth per PRB [MHz]
MAX_UE_THPT_MBPS = 1000.0 # Per-UE maximum throughput cap [Mbps]

# 3GPP-inspired MCS mapping: SINR thresholds [dB] -> Spectral efficiency [bps/Hz]
SINR_TH = np.array(
    [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0],
    dtype=np.float64,
)
EFF = np.array(
    [0.15, 0.23, 0.38, 0.60, 0.88, 1.18, 1.48, 1.91, 2.40, 2.73, 3.32, 3.90, 4.52, 5.12, 5.55],
    dtype=np.float64,
)

# Precomputed link-budget constant [dB]
LINK_CONST_DB = (
    P_TX_DBM
    - 10.0 * math.log10(TOTAL_PRB)
    + G_TX_DBI
    + G_RX_DBI
    - PEN_LOSS_DB
    - N_0_DBM
    - (32.44 + 20.0 * math.log10(FC_MHZ))
)

# =============================================================================
# 5. Traffic & Ground Area Configuration
# =============================================================================
LAMBDA_ARRIVAL = 0.0055
FIXED_FILE_SIZE_MB = 100.0

AREAS = [
    {"name": "Region-A", "pos": (35.18, 136.91), "weight": 0.32, "is_target": False},
    {"name": "Region-B", "pos": (34.69, 135.50), "weight": 0.31, "is_target": False},
    {"name": "Region-C", "pos": (33.59, 130.40), "weight": 0.20, "is_target": False},
    {"name": "Region-D", "pos": (34.38, 132.45), "weight": 0.11, "is_target": False},
    {"name": "Target",   "pos": (34.07, 134.55), "weight": 0.06, "is_target": True},
]
CENTER_LATLON = (34.0, 135.0)

# =============================================================================
# 6. Numerical Solver & Pruning Grid Parameters
# =============================================================================
SCAN_DT = 1.0
TIME_TOL = 1e-4
FINISH_EPS = 1e-6
GRID_DT = 1.0
VIS_ELEV_CENTER = 10.0
HEAP_INIT_PER_USER = 8
HEAP_HEADROOM = 8

# =============================================================================
# 7. Event Types & Baseline Policy Identifiers
# =============================================================================
EVENT_ARRIVAL = 1
EVENT_SAT_CHECK = 2
EVENT_PHYSICS = 3
EVENT_SIB_UPDATE = 4

M_BHO = 0       # Baseline Hard Handover (max SINR, 0 dB margin)
M_CHO = 1       # 3GPP-style Conditional Handover (max SINR, 3.0 dB margin)
M_LOAD_MIN = 2  # Load-Min (minimum connected users, 25 deg min elevation)
M_LOAD_STD = 3  # Load-Std (load-aware, 30 deg min elevation)

BASELINE_METHODS = (M_BHO, M_CHO, M_LOAD_MIN, M_LOAD_STD)

METHOD_LABELS = {
    M_BHO: "BHO",
    M_CHO: "Normal CHO",
    M_LOAD_MIN: "Load-Min",
    M_LOAD_STD: "Load-Std",
}
