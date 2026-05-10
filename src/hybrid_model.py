"""
hybrid_model.py — Optimized Hybrid DES + ABM Engine
Faithfully implements the granular flowchart architecture.
"""

import simpy
import numpy as np
import logging
from typing import NamedTuple

from src.environment import ServiceCenter
from src.agents import ServiceCenterABM, CustomerAgent
from src import config

log = logging.getLogger("hybrid_model")

def _evt(env, agent: CustomerAgent, event: str, stage: str = "—",
         wait: float = 0.0, service: float = 0.0) -> dict:
    return {
        "model": "Hybrid", "time": round(env.now, 2), "vehicle": agent.unique_id,
        "event": event, "stage": stage, "personality": agent.personality,
        "emotion": round(agent.emotion_val, 2), "patience": round(agent.patience_threshold, 1),
        "wait_min": round(wait, 2), "service_min": round(service, 2)
    }

def _patience_monitor(env: simpy.Environment, agent: CustomerAgent,
                      queue_start: float, target_process: simpy.Process):
    """
    Liu-Zhen Continuous Emotional Decay Patience Monitor.
    Event-driven intercept calculation.
    """
    try:
        while True:
            current_patience = agent.patience_threshold
            elapsed = env.now - queue_start
            time_left = current_patience - elapsed
            
            if time_left <= 0:
                if target_process.is_alive:
                    target_process.interrupt("patience_expired")
                return
            
            sleep_time = max(0.5, time_left / 2.0)
            yield env.timeout(sleep_time)
            agent.update_emotion_from_wait(env.now - queue_start)
            
    except simpy.Interrupt:
        pass  # Process finished or cancelled

def _customer_process(env: simpy.Environment, agent: CustomerAgent, sc: ServiceCenter, logs: list, rng_service: np.random.RandomState):
    """
    Executes the exact sequence defined in the Flowchart architecture.
    """
    # ── 1. Entering the parking lot ──
    agent.status = "Arrived"
    # Issue 4: Balking uses visible repair bays queue
    visible_q = sc.ql_general() + sc.ql_express()
    if agent.decide_balk(visible_q):
        logs.append(_evt(env, agent, "Balked", "Entry"))
        return
    logs.append(_evt(env, agent, "Arrived", "Parking Lot"))
    
    priority = 0 if (agent.personality == "Aggressive" and agent.emotion_val < 0.4) else 1
    t_start = env.now

    # ── 2. Wait for service advisor ──
    t_adv_q = env.now
    with sc.service_advisors.request(priority=priority) as req:
        monitor = env.process(_patience_monitor(env, agent, t_adv_q, env.active_process))
        try:
            yield req
            monitor.interrupt("served")
        except simpy.Interrupt:
            logs.append(_evt(env, agent, "Reneged", "Advisor", wait=env.now - t_adv_q))
            return

        wait_adv = env.now - t_adv_q
        svc_adv = config.get_advisor_time(rng_service)
        yield env.timeout(svc_adv)
        logs.append(_evt(env, agent, "AdvisorDone", "Advisor", wait=wait_adv, service=svc_adv))

    # ── 3. Vehicle inspection ──
    t_insp_q = env.now
    with sc.inspection_bays.request(priority=priority) as req:
        monitor = env.process(_patience_monitor(env, agent, t_insp_q, env.active_process))
        try:
            yield req
            monitor.interrupt("served")
        except simpy.Interrupt:
            logs.append(_evt(env, agent, "Reneged", "Inspection", wait=env.now - t_insp_q))
            return

        wait_insp = env.now - t_insp_q
        svc_insp = config.get_inspection_time(rng_service)
        yield env.timeout(svc_insp)
        logs.append(_evt(env, agent, "InspectionDone", "Inspection", wait=wait_insp, service=svc_insp))

    # ── 4. Waiting for a free bay ──
    is_express = rng_service.random() < config.EXPRESS_PROBABILITY
    bay_type = "Express" if is_express else "General"
    bay_res = sc.express_bays if is_express else sc.general_bays

    t_bay_q = env.now
    with bay_res.request(priority=priority) as req:
        monitor = env.process(_patience_monitor(env, agent, t_bay_q, env.active_process))
        try:
            yield req
            monitor.interrupt("served")
        except simpy.Interrupt:
            logs.append(_evt(env, agent, "Reneged", bay_type, wait=env.now - t_bay_q))
            return

        wait_bay = env.now - t_bay_q
        
        # ── 5. Drive the vehicle to bay ──
        drive_time = config.get_stage_time(rng_service, config.TIME_DRIVE_TO_BAY)
        yield env.timeout(drive_time)
        
        # ── 6. Clarify the scope of work ──
        clarify_time = config.get_stage_time(rng_service, config.TIME_CLARIFY_SCOPE)
        yield env.timeout(clarify_time)
        
        # ── 7. Able to carry out work? ──
        repair_svc_time = drive_time + clarify_time
        
        if rng_service.random() <= config.PROB_ABLE_TO_REPAIR:
            order_time = config.get_stage_time(rng_service, config.TIME_ORDER_REPAIR)
            yield env.timeout(order_time)
            parts_time = config.get_stage_time(rng_service, config.TIME_GET_SPARE_PARTS)
            yield env.timeout(parts_time)
            main_repair_time = config.get_service_time(rng_service, is_express)
            yield env.timeout(main_repair_time)
            check_time = config.get_stage_time(rng_service, config.TIME_CHECK_COMPLETED)
            yield env.timeout(check_time)
            repair_svc_time += (order_time + parts_time + main_repair_time + check_time)
            logs.append(_evt(env, agent, "RepairDone", bay_type, wait=wait_bay, service=repair_svc_time))
        else:
            # Issue 7: Repair fails -> log as RepairFailed but proceed to documentation
            logs.append(_evt(env, agent, "RepairFailed", bay_type, wait=wait_bay, service=repair_svc_time))

    # ── 8. Execute documentation work & pay ──
    doc_time = config.get_stage_time(rng_service, config.TIME_DOCUMENTATION_PAY)
    yield env.timeout(doc_time)
    
    # ── 9. OUT/END ──
    total_time = env.now - t_start
    logs.append(_evt(env, agent, "Departed", "Exit", wait=0.0, service=doc_time))
    logs.append(_evt(env, agent, "TotalTime", "All", service=total_time))

def _arrival_generator(env: simpy.Environment, sc: ServiceCenter, abm: ServiceCenterABM, logs: list, 
                       rng_arrival: np.random.RandomState, rng_service: np.random.RandomState, rng_agent: np.random.RandomState):
    """Arrivals via NHPP Thinning (Lewis-Shedler)"""
    while True:
        u_time = rng_arrival.exponential(1.0 / config.LAMBDA_MAX)
        yield env.timeout(u_time)
        exact_rate = config.get_arrival_rate(env.now)
        if rng_arrival.random() <= (exact_rate / config.LAMBDA_MAX):
            agent = abm.create_agent(rng_agent)
            env.process(_customer_process(env, agent, sc, logs, rng_service))

def run_hybrid(sim_time: float = config.SIMULATION_TIME, cfg: dict | None = None, seed: int = config.RANDOM_SEED) -> list[dict]:
    # Issue 2: Independent RNG streams for exact arrival/service matching between models
    rng_arrival = np.random.RandomState(seed)
    rng_service = np.random.RandomState(seed + 1)
    rng_agent = np.random.RandomState(seed + 2)
    
    logs: list[dict] = []
    env = simpy.Environment()
    sc = ServiceCenter(env, cfg)
    abm = ServiceCenterABM()

    env.process(_arrival_generator(env, sc, abm, logs, rng_arrival, rng_service, rng_agent))
    env.run(until=sim_time)
    return logs
