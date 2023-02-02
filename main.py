import argparse
from pathlib import Path
import time
import yaml
from logger import Logger
from network import Network
import pandas as pd
from runner import Runner

CONFIG_PATH = Path("./config/")

def load_config(filename):
    """Load yaml configuration file.

    Args:
        filename (str): config file name (including extension).

    Returns:
        dict: configuration dictionary.
    """
    with open(Path(CONFIG_PATH / filename)) as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transport network and School Choice")
    parser.add_argument('--config', default='default.yaml', type=str)
    parser.add_argument('--no_log', action='store_true', default=False)

    args = parser.parse_args()

    config = load_config(args.config)

    sim_start = time.time()
    if not args.no_log:
        logger = Logger(config)
        print(f'Running {args.config} configuration - results saved in {logger.results_path}')
    else:
        logger = None
        print(f'Running {args.config} configuration - no results saved')
        

    # TODO: probably want to move calc_tt_mx flag to config file (maybe want to set to False for large networks.)
    network = Network(config['network_file'], calc_tt_mx=True)
    population = pd.read_csv(config['population_file'])
    facilities = pd.read_csv(config['facilities_file'])

    runner = Runner(network, population, facilities, logger)

    runner.run_simulation(
            config['simulation_rounds'],
            config['allocation_rounds'],
            config['intervention_rounds'],
            config['intervention_budget'],
            config['preferences_model'],
            config['allocation_model'], 
            config['intervention_model'], 
            nearest_k_k=config.get('nearest_k', None),
            update_preference_params=config['update_preference_params'])
    
    sim_time = time.time() - sim_start
    print(f"All is said and done in {sim_time} seconds, which is {sim_time / 60} minutes.")
    
    if logger:
        logger.append_to_output_file(f"sim_time: {sim_time}")
