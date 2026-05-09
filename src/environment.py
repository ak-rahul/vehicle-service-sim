"""
environment.py — Physical SimPy resource containers.
Optimized with PriorityResource for better throughput handling.
"""

import simpy
from src import config

class ServiceCenter:
    """
    Holds all SimPy PriorityResources representing the physical service center.
    Allows for prioritization in queues (e.g., highly agitated customers could 
    theoretically be escalated).
    """
    def __init__(self, env: simpy.Environment, cfg: dict | None = None):
        self.env = env
        cfg = cfg or {}

        n_adv  = cfg.get("num_advisors",   config.NUM_SERVICE_ADVISORS)
        n_insp = cfg.get("num_inspection", config.NUM_INSPECTION_BAYS)
        n_gen  = cfg.get("num_general",    config.NUM_GENERAL_BAYS)
        n_exp  = cfg.get("num_express",    config.NUM_EXPRESS_BAYS)

        # ── Priority Resources ──
        # Priority 0 is highest, 1 is normal, etc.
        self.service_advisors  = simpy.PriorityResource(env, capacity=n_adv)
        self.inspection_bays   = simpy.PriorityResource(env, capacity=n_insp)
        self.general_bays      = simpy.PriorityResource(env, capacity=n_gen)
        self.express_bays      = simpy.PriorityResource(env, capacity=n_exp)

    def ql_advisor(self)    -> int: return len(self.service_advisors.queue)
    def ql_inspection(self) -> int: return len(self.inspection_bays.queue)
    def ql_general(self)    -> int: return len(self.general_bays.queue)
    def ql_express(self)    -> int: return len(self.express_bays.queue)
