import argparse
from pathlib import Path
import yaml
from network import Network
import pandas as pd
from preference import nearest_k
from allocation import identity
from evaluation import facility_capacity

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

    # Calculate travel times for all agents in the population to all facilities.
    tt = network.tt_mx[population['node'].values][:, [facilities['node'].values]].squeeze()

    # Just a random check if the above indexing works, to remove later on.
    pop_sample = population.sample()
    fac_sample = facilities.sample()
    assert network.tt_mx[pop_sample.iloc[0]['node'], fac_sample.iloc[0]['node']] == tt[pop_sample.iloc[0]['id'], fac_sample.iloc[0]['id']], "Something wrong with travel time indexing - incompatible travel times between network pre-calculated and indexed values."

    # Generate preference list for each agent.
    pref_list = nearest_k(tt, k=1)
    allocation = identity(pref_list)
    capacity_eval = facility_capacity(population, facilities, allocation)

    print(f'Facility capacity evaluation: {capacity_eval}')
