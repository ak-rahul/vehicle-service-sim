import argparse
import logging
import json

from src import config
from src.des_model import run_des
from src.hybrid_model import run_hybrid
from src.analysis import extract_kpis, export_static_charts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SimulationRunner")

def print_banner():
    banner = """
    ========================================================
     [VEHICLE] SERVICE CENTER SIMULATION
        (Pure DES vs. Hybrid ABM)
    ========================================================
    """
    print(banner)

def main():
    parser = argparse.ArgumentParser(description="Run the Vehicle Service Center Simulation.")
    parser.add_argument('--time', type=int, default=config.SIMULATION_TIME, 
                        help='Total simulation time in minutes')
    parser.add_argument('--advisors', type=int, default=config.NUM_SERVICE_ADVISORS, 
                        help='Number of service advisors')
    parser.add_argument('--bays', type=int, default=config.NUM_GENERAL_BAYS, 
                        help='Number of general service bays available')
    parser.add_argument('--express', type=int, default=config.NUM_EXPRESS_BAYS, 
                        help='Number of express service bays available')
    parser.add_argument('--inspection', type=int, default=config.NUM_INSPECTION_BAYS, 
                        help='Number of inspection bays')
    parser.add_argument('--mode', choices=['des', 'hybrid', 'both'], default='both',
                        help='Which model to run')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed debug logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    cfg_override = {
        "num_advisors": args.advisors,
        "num_inspection": args.inspection,
        "num_general": args.bays,
        "num_express": args.express
    }

    print_banner()
    logger.info(f"Initializing Environment parameters:")
    logger.info(f"- Simulation Time:     {args.time} minutes")
    logger.info(f"- Advisors:            {args.advisors}")
    logger.info(f"- Inspection Bays:     {args.inspection}")
    logger.info(f"- General Bays:        {args.bays}")
    logger.info(f"- Express Bays:        {args.express}")
    
    des_logs = []
    hybrid_logs = []
    des_kpis = {}
    hybrid_kpis = {}
    
    seed = config.RANDOM_SEED
    
    if args.mode in ['des', 'both']:
        logger.info("Starting Pure DES simulation...")
        des_logs = run_des(sim_time=args.time, cfg=cfg_override, seed=seed)
        des_kpis = extract_kpis(des_logs, "DES", cfg=cfg_override, sim_time=args.time)
        print("\n--- DES KPIs ---")
        print(json.dumps(des_kpis, indent=2))
        
    if args.mode in ['hybrid', 'both']:
        logger.info("Starting Hybrid simulation...")
        hybrid_logs = run_hybrid(sim_time=args.time, cfg=cfg_override, seed=seed)
        hybrid_kpis = extract_kpis(hybrid_logs, "Hybrid", cfg=cfg_override, sim_time=args.time)
        print("\n--- Hybrid KPIs ---")
        print(json.dumps(hybrid_kpis, indent=2))
        
    if args.mode == 'both':
        logger.info("Generating comparison charts in outputs/ folder...")
        export_static_charts(des_kpis, hybrid_kpis, des_logs, hybrid_logs, "outputs")
        
    print("\n[Done] Simulation sequence completed!")

if __name__ == "__main__":
    main()
