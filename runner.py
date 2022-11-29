from logger import Logger
import pandas as pd
from allocation import first_choice, random_serial_dictatorship
from evaluation import facility_capacity, facility_diversity
from intervention import create_random_edge
import matplotlib.pyplot as plt
from network import Network
from preference import toy_model, nearest_k

class Runner(object):
    def __init__(self, network: Network, population: pd.DataFrame, facilities: pd.DataFrame, logger: Logger):

        self.network = network
        self.population = population
        self.facilities = facilities
        self.logger = logger

        # Calculate travel times for all agents in the population to all facilities.
        self.travel_time = self.network.tt_mx[self.population['node'].values][:, [self.facilities['node'].values]].squeeze()

        ## Just a random check if the above indexing works, to remove later on.
        pop_sample = population.sample()
        fac_sample = facilities.sample()
        assert self.network.tt_mx[pop_sample.iloc[0]['node'], fac_sample.iloc[0]['node']] == self.travel_time[pop_sample.iloc[0]['id'], fac_sample.iloc[0]['id']], "Something wrong with travel time indexing - incompatible travel times between network pre-calculated and indexed values."
        ##

    def run_simulation(self, simulation_rounds: int, preferences_model: str, allocation_model: str, intervention_model: str, nearest_k_k=None):
        """Runs a simulation of specified simulation_rounds using specified preferences, allocation and intervention models.

        Args:
            simulation_rounds (int): total nr of simulation rounds to run.
            preferences_model (str): preference model to use.
            allocation_model (str): allocation model to use.
            intervention_model (str): network intervention model to use.
            nearest_k_k (int, optional): k parameter in nearest_k preference model. Defaults to None.
        """

        # TODO - maybe replace with scenario builder.
        # Note: first round is vanilla - no interventions are added.
        _, _, capacity_eval, diversity_eval = self.run_agent_round(preferences_model, allocation_model, nearest_k_k)
        capacity = []
        diversity = []
        for i in range(simulation_rounds):
            self.create_intervention(intervention_model)

            _, _, capacity_eval, diversity_eval = self.run_agent_round(preferences_model, allocation_model, nearest_k_k)
            capacity.append(capacity_eval)
            diversity.append(diversity_eval)

            print(f'Facility capacity evaluation: {capacity_eval}')
            print(f'Facility diversity evaluation: fac1: {diversity_eval[0][0]} - {diversity_eval[0][1]}, fac2: {diversity_eval[1][0]} - {diversity_eval[1][1]}')

        # Generate group composition plot for every facility.
        for fid in range(self.facilities.shape[0]):
            grp0_pct = []
            grp1_pct = []
            for i in range(simulation_rounds):
                grp0_pct.append(diversity[i][fid][0])
                grp1_pct.append(diversity[i][fid][1])
            
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(range(simulation_rounds), grp0_pct, label='grp0')
            ax.plot(range(simulation_rounds), grp1_pct, label='grp1')
            ax.set_ylim(0, 1)
            ax.fill_between(range(simulation_rounds), grp0_pct, grp1_pct, color='#E8E8E8')
            ax.hlines(y=0.5, xmin=0, xmax=simulation_rounds-1, color='gray', linestyle='--')
            ax.set_title(f"Facility {self.facilities.iloc[fid]['facility']} ({fid}) - Group Composition")
            ax.legend()

            if self.logger:
                self.logger.save_plot(fig, f'facility_{fid}_group_composition.png')

    def run_agent_round(self, preferences_model, allocation_model, nearest_k_k=None):
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

        assert pref_list is not None, 'No preference list was generated, specify a valid preferences_model parameter in config.'
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
        
        assert allocation is not None, 'No allocation list was generated, specify a valid allocation_model parameter in config.'
        return allocation
    
    def create_intervention(self, intervention_model: str):
        """Creates and adds an intervention (new edge) to the network, according to the intervention_model

        Args:
            intervention_model (str): network intervention model to use.
        """
        x, y, w = None, None, None
        if intervention_model == 'random':
            x, y, w = create_random_edge(self.network)

        assert x is not None, 'No intervention was generated, specify a valid intervention_model parameter in config.'

        print(f'adding ({x}, {y}) edge')
        self.network.add_edge(x, y, w)

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
