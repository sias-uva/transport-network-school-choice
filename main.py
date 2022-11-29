import argparse
from pathlib import Path
import time
import yaml
from intervention import create_random_edge
from network import Network
import pandas as pd
from runner import Runner
import datetime

## TODO - Automatically copy the config file to the output folder.

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

    # Create output folder.
    now = datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')
    save_dir = Path('./results') / f"{now}_{config['preferences_model']}_{config['allocation_model']}_{config['intervention_model']}"
    save_dir.mkdir(parents=True, exist_ok=True)

    sim_start = time.time()
    print(f'Running on {args.config} configuration - results saved in {save_dir}')
    
    if not args.no_log:
        with open(save_dir / 'config.txt', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    # TODO: probably want to move calc_tt_mx flag to config file (maybe want to set to False for large networks.)
    network = Network(config['network_file'], calc_tt_mx=True)
    population = pd.read_csv(config['population_file'])
    facilities = pd.read_csv(config['facilities_file'])

    runner = Runner(network, population, facilities)

    runner.run_simulation(
            config['simulation_rounds'],
            config['preferences_model'], 
            config['allocation_model'], 
            config['intervention_model'], 
            config.get('nearest_k', None))
    
    sim_time = time.time() - sim_start
    print(f"All is said and done in {sim_time} seconds, which is {sim_time / 60} minutes.")
    if not args.no_log:
        with open(save_dir / 'config.txt', 'a') as f:
            f.write(f"sim_time: {sim_time}")
