"""
des_model.py — Optimized Pure DES Engine
Faithfully implements the granular flowchart architecture without ABM dynamic traits,
but uses comparable patience mechanics for apple-to-apple comparisons.
"""

import simpy
import numpy as np
import logging
from typing import NamedTuple

from src.environment import ServiceCenter
from src import config

log = logging.getLogger("des_model")

class LogEvent(NamedTuple):
    model: str
    time: float
    vehicle: int
    event: str
    stage: str
    wait_min: float
    service_min: float

def _evt(env, vid: int, event: str, stage: str = "—", wait: float = 0.0, service: float = 0.0) -> dict:
    return LogEvent(
        model="DES", time=round(env.now, 2), vehicle=vid,
        event=event, stage=stage,
        wait_min=round(wait, 2), service_min=round(service, 2)
    )._asdict()

def _vehicle_process(env: simpy.Environment, vid: int, sc: ServiceCenter, logs: list, 
                     rng_service: np.random.RandomState, rng_agent: np.random.RandomState):
    """Executes the exact sequence defined in the Flowchart architecture."""
    # Issue 5: Use a sampled base patience for DES to make it comparable to Hybrid
    personality = rng_agent.choice(["Conservative", "Steady", "Aggressive"], p=config.PERSONALITY_PROBS)
    mu, sig = config.PATIENCE_PARAMS[personality]
    patience = rng_agent.lognormal(mu, sig)
    balk_threshold = config.BALK_THRESHOLDS[personality]
    
    # ── 1. Entering the parking lot ──
    # Issue 4: Balking is wired to the visible bay queues
    visible_q = sc.ql_general() + sc.ql_express()
    if visible_q >= balk_threshold:
        logs.append(_evt(env, vid, "Balked", "Entry"))
        return
    logs.append(_evt(env, vid, "Arrived", "Parking Lot"))
    
    priority = 1 # Normal priority
    t_start = env.now

    # ── 2. Wait for service advisor ──
    t_adv_q = env.now
    with sc.service_advisors.request(priority=priority) as req:
        # We use fixed patience per stage or total? In DES usually it's per stage timeout, 
        # but to match hybrid we will use the generated base patience as the timeout.
        result = yield req | env.timeout(patience)
        if req not in result:
            logs.append(_evt(env, vid, "Reneged", "Advisor", wait=env.now - t_adv_q))
            return
            
        wait_adv = env.now - t_adv_q
        svc_adv = config.get_advisor_time(rng_service)
        yield env.timeout(svc_adv)
        logs.append(_evt(env, vid, "AdvisorDone", "Advisor", wait=wait_adv, service=svc_adv))

    # ── 3. Vehicle inspection ──
    t_insp_q = env.now
    with sc.inspection_bays.request(priority=priority) as req:
        result = yield req | env.timeout(patience)
        if req not in result:
            logs.append(_evt(env, vid, "Reneged", "Inspection", wait=env.now - t_insp_q))
            return
            
        wait_insp = env.now - t_insp_q
        svc_insp = config.get_inspection_time(rng_service)
        yield env.timeout(svc_insp)
        logs.append(_evt(env, vid, "InspectionDone", "Inspection", wait=wait_insp, service=svc_insp))

    # ── 4. Waiting for a free bay ──
    is_express = rng_service.random() < config.EXPRESS_PROBABILITY
    bay_type = "Express" if is_express else "General"
    bay_res = sc.express_bays if is_express else sc.general_bays

    t_bay_q = env.now
    with bay_res.request(priority=priority) as req:
        result = yield req | env.timeout(patience)
        if req not in result:
            logs.append(_evt(env, vid, "Reneged", bay_type, wait=env.now - t_bay_q))
            return
            
        wait_bay = env.now - t_bay_q
        
        # ── 5. Drive the vehicle to bay ──
        yield env.timeout(config.TIME_DRIVE_TO_BAY)
        
        # ── 6. Clarify the scope of work ──
        yield env.timeout(config.TIME_CLARIFY_SCOPE)
        
        # ── 7. Able to carry out work? ──
        repair_svc_time = config.TIME_DRIVE_TO_BAY + config.TIME_CLARIFY_SCOPE
        
        if rng_service.random() <= config.PROB_ABLE_TO_REPAIR:
            yield env.timeout(config.TIME_ORDER_REPAIR)
            yield env.timeout(config.TIME_GET_SPARE_PARTS)
            main_repair_time = config.get_service_time(rng_service, is_express)
            yield env.timeout(main_repair_time)
            yield env.timeout(config.TIME_CHECK_COMPLETED)
            repair_svc_time += (config.TIME_ORDER_REPAIR + config.TIME_GET_SPARE_PARTS + main_repair_time + config.TIME_CHECK_COMPLETED)
            logs.append(_evt(env, vid, "RepairDone", bay_type, wait=wait_bay, service=repair_svc_time))
        else:
            logs.append(_evt(env, vid, "RepairFailed", bay_type, wait=wait_bay, service=repair_svc_time))

    # ── 8. Execute documentation work & pay ──
    yield env.timeout(config.TIME_DOCUMENTATION_PAY)
    
    # ── 9. OUT/END ──
    total_time = env.now - t_start
    logs.append(_evt(env, vid, "Departed", "Exit", wait=0.0, service=config.TIME_DOCUMENTATION_PAY))
    logs.append(_evt(env, vid, "TotalTime", "All", service=total_time))

def _arrival_generator(env: simpy.Environment, sc: ServiceCenter, logs: list,
                       rng_arrival: np.random.RandomState, rng_service: np.random.RandomState, rng_agent: np.random.RandomState):
    """Arrivals via NHPP Thinning (Lewis-Shedler)"""
    vid = 1
    while True:
        u_time = rng_arrival.exponential(1.0 / config.LAMBDA_MAX)
        yield env.timeout(u_time)
        exact_rate = config.get_arrival_rate(env.now)
        if rng_arrival.random() <= (exact_rate / config.LAMBDA_MAX):
            env.process(_vehicle_process(env, vid, sc, logs, rng_service, rng_agent))
            vid += 1

def run_des(sim_time: float = config.SIMULATION_TIME, cfg: dict | None = None, seed: int = config.RANDOM_SEED) -> list[dict]:
    # Use exact same streams to guarantee 100% identical arrival times and baseline attributes
    rng_arrival = np.random.RandomState(seed)
    rng_service = np.random.RandomState(seed + 1)
    rng_agent = np.random.RandomState(seed + 2)
    
    logs: list[dict] = []
    env = simpy.Environment()
    sc = ServiceCenter(env, cfg)
    env.process(_arrival_generator(env, sc, logs, rng_arrival, rng_service, rng_agent))
    env.run(until=sim_time)
    return logs
