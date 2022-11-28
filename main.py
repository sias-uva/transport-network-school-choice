import argparse
from pathlib import Path
import yaml
from intervention import create_random_edge
from network import Network
import pandas as pd
from runner import Runner

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

    args = parser.parse_args()

    print(f'Running on {args.config} configuration...')

    config = load_config(args.config)

    # TODO: probably want to move calc_tt_mx flag to config file (maybe want to set to False for large networks.)
    network = Network(config['network_file'], calc_tt_mx=True)
    population = pd.read_csv(config['population_file'])
    facilities = pd.read_csv(config['facilities_file'])

    runner = Runner(network, population, facilities)

    _, _, capacity_eval, diversity_eval = runner.run_round(config['preferences_model'], config['allocation_model'], config.get('nearest_k', None))
    print(f'Facility capacity evaluation: {capacity_eval}')
    print(f'Facility diversity evaluation: fac1: {diversity_eval[0][0]} - {diversity_eval[0][1]}, fac2: {diversity_eval[1][0]} - {diversity_eval[1][1]}')

    # Add 5 random edges
    # TODO - maybe replace with scenario builder.
    for i in range(5):
        x, y, w = create_random_edge(network)
        print(f'adding ({x}, {y}) edge')
        network.add_edge(x, y, w)

        _, _, capacity_eval, diversity_eval = runner.run_round(config['preferences_model'], config['allocation_model'], config.get('nearest_k', None))
        print(f'Facility capacity evaluation: {capacity_eval}')
        print(f'Facility diversity evaluation: fac1: {diversity_eval[0][0]} - {diversity_eval[0][1]}, fac2: {diversity_eval[1][0]} - {diversity_eval[1][1]}')
