import pandas as pd
from allocation import first_choice, random_serial_dictatorship
from evaluation import facility_capacity, facility_diversity

from network import Network
from preference import toy_model, nearest_k

class Runner(object):
    def __init__(self, network: Network, population: pd.DataFrame, facilities: pd.DataFrame):

        self.network = network
        self.population = population
        self.facilities = facilities

        # Calculate travel times for all agents in the population to all facilities.
        self.travel_time = self.network.tt_mx[self.population['node'].values][:, [self.facilities['node'].values]].squeeze()

        ## Just a random check if the above indexing works, to remove later on.
        pop_sample = population.sample()
        fac_sample = facilities.sample()
        assert network.tt_mx[pop_sample.iloc[0]['node'], fac_sample.iloc[0]['node']] == self.travel_time[pop_sample.iloc[0]['id'], fac_sample.iloc[0]['id']], "Something wrong with travel time indexing - incompatible travel times between network pre-calculated and indexed values."
        ##

    def run_round(self, preferences_model, allocation_model, nearest_k_k=None):
        """Runs a round of preference generation -> allocation generation -> evaluation.

        Args:
            preferences_model (str): model to use to generate preferences
            allocation_model (str): model to use to generate allocations
            nearest_k_k (int, optional): k parameter in nearest_k preference model. Defaults to None.

        Returns:
            list: preference_list, allocation, capacity_eval, diversity_eval
        """
        pref_list = self.generate_preferences(preferences_model, nearest_k_k=nearest_k_k)
        allocation = self.generate_allocation(pref_list, allocation_model)
        capacity_eval, diversity_eval = self.evaluate(allocation)

        return pref_list, allocation, capacity_eval, diversity_eval

    def generate_preferences(self, preferences_model: str, nearest_k_k=None):
        """Generates preferences for each agent in the population, according to preferences_model.

        Args:
            preferences_model (str): preference model to use.
            nearest_k_k (int, optional): k parameter in nearest_k preference model. Defaults to None.

        Returns:
            np.array: array of size (nr of agents, nr of facilities) where each facility is sorted by preference.
        """
        pref_list = None
        if preferences_model == 'nearest_k':
            assert nearest_k_k, 'You need to specify nearest_k parameter in config.'
            pref_list = nearest_k(self.travel_time, k=nearest_k_k)
        elif preferences_model == 'toy_model':
            # Select facility qualities
            qualities = self.facilities.quality.to_numpy()
            pref_list = toy_model(self.travel_time, qualities)

        assert pref_list is not None, 'No preference list was generated, specify preferences_model parameter in config.'
        return pref_list
    
    def generate_allocation(self, pref_list, allocation_model):
        """Generates allocation of facilities to agents according to allocation_model.

        Args:
            pref_list (np.array): array of size (nr of agents, nr of facilities) where each facility is sorted by preference.
            allocation_model (str): allocation model to use.

        Returns:
            np.array: array of size  (nr_agents, 1) where each agent is assigned to one facility.
        """
        # Assign agents to facilities using an allocation model.
        allocation = None
        if allocation_model == 'first_choice':
            allocation = first_choice(pref_list)
        elif allocation_model == 'random_serial_dictatorship':
            capacities = self.facilities.capacity.copy().to_numpy()
            allocation = random_serial_dictatorship(pref_list, capacities)
        
        assert allocation is not None, 'No allocation list was generated, specify allocation_model parameter in config.'
        return allocation
    
    def evaluate(self, allocation):
        """Evaluates allocation according to evaluation metrics.

        Args:
            allocation (np.array): array of size  (nr_agents, 1) where each agent is assigned to one facility.

        Returns:
            list: list of evaluation metrics.
        """
        capacity_eval = facility_capacity(self.population, self.facilities, allocation)
        diversity_eval = facility_diversity(self.population, self.facilities, allocation)

        return capacity_eval, diversity_eval
