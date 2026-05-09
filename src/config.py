"""
Central Configuration Module — Vehicle Service Center Simulation
Optimized with rigorous statistical distributions, NHPP, and full Flowchart granularity.
"""

import math
import numpy as np
from typing import Tuple

# ─────────────────────────────────────────────
#  General Simulation Settings
# ─────────────────────────────────────────────
SIMULATION_TIME   = 600   # minutes
RANDOM_SEED       = 42

# ─────────────────────────────────────────────
#  Facility Resources
# ─────────────────────────────────────────────
NUM_SERVICE_ADVISORS  = 2
NUM_INSPECTION_BAYS   = 2
NUM_GENERAL_BAYS      = 5
NUM_EXPRESS_BAYS      = 1

# ─────────────────────────────────────────────
#  Arrival Rates (Vehicles / min) - From Architecture 03
# ─────────────────────────────────────────────
# Original lambda = 0.019 veh/min, but we apply NHPP thinning for rush hours.
LAMBDA_BASE = 0.019
LAMBDA_MAX  = LAMBDA_BASE * 3.0 

# ─────────────────────────────────────────────
#  Service Time Distributions (minutes)
#  Using Lognormal for realistic right-skewed non-negative times.
# ─────────────────────────────────────────────
def _lognorm_params(mean: float, std: float) -> Tuple[float, float]:
    """Converts arithmetic mean & std to lognormal mu & sigma parameters."""
    sigma2 = math.log(1 + (std**2 / mean**2))
    mu = math.log(mean) - (sigma2 / 2)
    return mu, math.sqrt(sigma2)

MU_ADV, SIG_ADV   = _lognorm_params(5.0, 1.5)
MU_INSP, SIG_INSP = _lognorm_params(10.0, 2.0)

# The total mu=166.7 from the architecture is broken down into granular stages
MU_GEN, SIG_GEN   = _lognorm_params(120.0, 20.0)
MU_EXP, SIG_EXP   = _lognorm_params(30.0, 5.0)

# Granular Flowchart Stages (Deterministic or tightly bound normals)
TIME_DRIVE_TO_BAY      = 1.0
TIME_CLARIFY_SCOPE     = 3.0
TIME_ORDER_REPAIR      = 2.0
TIME_GET_SPARE_PARTS   = 10.0
TIME_CHECK_COMPLETED   = 5.0
TIME_DOCUMENTATION_PAY = 5.0

PROB_ABLE_TO_REPAIR    = 0.95  # 5% chance the center cannot do the work (no parts/tools)
EXPRESS_PROBABILITY    = 0.20

# ─────────────────────────────────────────────
#  DES-Only Model Parameters
# ─────────────────────────────────────────────
DES_MAX_WAIT_ADVISOR    = 45.0
DES_MAX_WAIT_INSPECTION = 60.0
DES_MAX_WAIT_BAY        = 120.0
DES_BALK_THRESHOLD      = 8

# ─────────────────────────────────────────────
#  Hybrid / ABM Agent Parameters
#  Implementation of the Liu-Zhen Emotional Contagion/Decay Model
# ─────────────────────────────────────────────
PATIENCE_PARAMS = {
    "Conservative": _lognorm_params(90.0, 15.0),
    "Steady":       _lognorm_params(67.5, 11.25),
    "Aggressive":   _lognorm_params(30.0, 7.5),
}

BALK_THRESHOLDS = {
    "Conservative": 10,
    "Steady":        6,
    "Aggressive":    3,
}

PERSONALITY_PROBS = [0.20, 0.50, 0.30]
# Initial emotion values [0.0 = terrible, 1.0 = fantastic]
EMOTION_INITIAL_MU = 0.8
EMOTION_INITIAL_SIGMA = 0.1

# ─────────────────────────────────────────────
#  Optimization Sweep Ranges
# ─────────────────────────────────────────────
OPT_ADVISORS_RANGE    = [1, 2, 3]
OPT_INSPECTION_RANGE  = [1, 2, 3]
OPT_GENERAL_RANGE     = [4, 5, 6, 7]
OPT_EXPRESS_RANGE     = [1, 2]

# ─────────────────────────────────────────────
#  Stochastic Sampling Helpers
# ─────────────────────────────────────────────
def get_arrival_rate(t: float) -> float:
    """Returns exact arrival rate lambda(t) for NHPP."""
    t_mod = t % 1440
    if (0 <= t_mod <= 120) or (300 <= t_mod <= 420):
        return LAMBDA_BASE * 3.0
    elif (120 < t_mod <= 180) or (240 <= t_mod < 300):
        return LAMBDA_BASE * 1.5
    return LAMBDA_BASE

def get_advisor_time() -> float: return np.random.lognormal(MU_ADV, SIG_ADV)
def get_inspection_time() -> float: return np.random.lognormal(MU_INSP, SIG_INSP)
def get_service_time(is_express: bool = False) -> float:
    if is_express: return np.random.lognormal(MU_EXP, SIG_EXP)
    return np.random.lognormal(MU_GEN, SIG_GEN)
