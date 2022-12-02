import numpy as np
from logger import Logger
import pandas as pd
from allocation import first_choice, random_serial_dictatorship
from evaluation import dissimilarity_index, facility_capacity, facility_group_composition, facility_rank_distribution
from intervention import create_random_edge
import matplotlib

from plot import heatmap_from_numpy
# Matplotlib stopped working on my machine, so I had to add this line to make it work again.
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from network import Network
from preference import toy_model, nearest_k

class Runner(object):
    def __init__(self, network: Network, population: pd.DataFrame, facilities: pd.DataFrame, logger: Logger):

        self.network = network
        self.population = population
        self.facilities = facilities
        self.logger = logger

        self.facilities_size = facilities.shape[0]
        self.population_size = population.shape[0]
        self.groups= population['group'].unique()
        self.total_groups = population['group'].nunique()
        # Log stuff
        logger.append_to_output_file(f'facilities_size: {self.facilities_size}\npopulation_size: {self.population_size}\ntotal_groups: {self.total_groups}')
        for g in self.groups:
            logger.append_to_output_file(f"Group {g} size: {population[population['group'] == g].shape[0]}")

        # Calculate travel times for all agents in the population to all facilities.
        self.travel_time = self.network.tt_mx[self.population['node'].values][:, [self.facilities['node'].values]].squeeze()

        ## Just a random check if the above indexing works, to remove later on.
        pop_sample = population.sample()
        fac_sample = facilities.sample()
        assert self.network.tt_mx[pop_sample.iloc[0]['node'], fac_sample.iloc[0]['node']] == self.travel_time[pop_sample.iloc[0]['id'], fac_sample.iloc[0]['id']], "Something wrong with travel time indexing - incompatible travel times between network pre-calculated and indexed values."
        ##

        # Safe a figure of the network to the output folder.
        if self.logger:
            self.logger.save_igraph_plot(self.network)

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
        # Note: this currently only runs properly for 2 groups.

        # Note: first round is vanilla - no interventions are added.
        _, _, eval_metrics = self.run_agent_round(preferences_model, allocation_model, nearest_k_k)
        # initialize empty numpy arrays meant to store values of evaluation metrics per simulation round.
        alloc_by_facility = np.zeros((simulation_rounds, self.facilities_size))
        capacity = np.zeros((simulation_rounds, self.facilities_size))
        grp_composition_pct = np.zeros((simulation_rounds, self.facilities_size, self.total_groups))
        grp_composition = np.zeros((simulation_rounds, self.facilities_size, self.total_groups))
        dissimilarity_index = np.zeros(simulation_rounds)

        for i in range(simulation_rounds):
            intervention = self.create_intervention(intervention_model)

            _, _, eval_metrics = self.run_agent_round(preferences_model, allocation_model, nearest_k_k)
            alloc_by_facility[i] = eval_metrics['alloc_by_facility']
            capacity[i] = eval_metrics['capacity']
            grp_composition_pct[i] = eval_metrics['grp_composition_pct']
            grp_composition[i] = eval_metrics['grp_composition']
            dissimilarity_index[i] = eval_metrics['dissimilarity_index']

            alloc_heatmap = heatmap_from_numpy(eval_metrics['grp_composition'], 
                                    title=f"Allocation by facility and group - round {i}", 
                                    subtitle=f"{preferences_model} - {allocation_model} - {intervention_model}",
                                    xlabel='Groups',
                                    ylabel='Facilities')
            if self.logger:
                self.logger.save_plot(alloc_heatmap, f"allocation_by_facility_and_group_{i}.png", round=i)

            rank_distribution_heatmap = heatmap_from_numpy(eval_metrics['facility_rank_distr'], 
                                        title=f"Facility rank distribution - round {i}",
                                        subtitle=f"{preferences_model} - {allocation_model} - {intervention_model}",
                                        xlabel='Rank',
                                        ylabel='Facilities')
            if self.logger:
                self.logger.save_plot(rank_distribution_heatmap, f"rank_distribution_{i}.png", round=i)


            # pref_heatmap = heatmap_from_numpy(eval_metrics['pref_by_facility'],)

            # Create a plot with the network intervention.
            if self.logger:
                self.logger.save_igraph_plot(self.network, f"intervention_{i}.pdf", edges_to_color=intervention , round=i)
            # fig = self.network.create_network_figure(self.logger.rounds_path / str(i) / f"intervention_{i}.pdf", edges_to_color=[intervention])

        # Generate group composition plot for each facility (diffrent plots).
        for fid in range(self.facilities_size):
            fig, ax = plt.subplots(figsize=(5, 5))
            for gid in range(self.total_groups):
                ax.plot(range(simulation_rounds), grp_composition_pct[:, fid, gid], label=f'Group {gid}')
            
            # ax.fill_between(range(simulation_rounds), grp0_pct, grp1_pct, color='#E8E8E8') # commented out because it does not generalize to more than 2 groups.
            ax.set_xlabel('Simulation round')
            ax.set_ylabel('Group composition')
            ax.set_ylim(0, 1)
            ax.hlines(y=0.5, xmin=0, xmax=simulation_rounds-1, color='gray', linestyle='--')
            fig.suptitle(f"Facility {self.facilities.iloc[fid]['facility']} ({fid}) - Group Composition")
            ax.set_title(f"{preferences_model} - {allocation_model} - {intervention_model}")
            ax.legend()

            if self.logger:
                self.logger.save_plot(fig, f'facility_{fid}_group_composition.png')

        # Generate Dissimilarity Index plot for all facilities.
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(range(simulation_rounds), dissimilarity_index, label=f'Dissimilarity Index')
        ax.set_xlabel('Simulation round')
        ax.set_ylabel('Dissimilarity Index')
        ax.set_ylim(0, 1)
        fig.suptitle(f"Dissimilarity Index")
        ax.set_title(f"{preferences_model} - {allocation_model} - {intervention_model}")
        if self.logger:
            self.logger.save_plot(fig, f'dissimilarity_index.png')
        
        # Generate Capacity plot for all facilities.
        fig, ax = plt.subplots(figsize=(5, 5))
        for fid in range(self.facilities_size):
            ax.plot(range(simulation_rounds), capacity[:, fid], label=f'Facility {fid}')
        ax.set_xlabel('Simulation round')
        ax.set_ylabel('Capacity %')
        ax.set_ylim(0, 2)
        ax.legend()
        fig.suptitle(f"Facility Capacity")
        ax.set_title(f"{preferences_model} - {allocation_model} - {intervention_model}")
        if self.logger:
            self.logger.save_plot(fig, f'facility_capacity.png')

        # Generate plot of all network interventions
        if self.logger:
            self.logger.save_igraph_plot(self.network, f"network_interventions.pdf", edges_to_color=self.network.added_edges)
        
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
        eval_metrics = self.evaluate(pref_list, allocation)

        return pref_list, allocation, eval_metrics

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
        return self.network.add_edge(x, y, w)

    def evaluate(self, pref_list, allocation):
        """Evaluates allocation according to evaluation metrics.

        Args:
            allocation (np.array): array of size  (nr_agents, 1) where each agent is assigned to one facility.

        Returns:
            dit: dictionary of evaluation metrics.
        """
        alloc_by_facility, capacity = facility_capacity(self.population, self.facilities, allocation)
        grp_composition, grp_composition_pct = facility_group_composition(self.population, self.facilities, allocation)
        facility_rank_distr = facility_rank_distribution(pref_list, self.facilities_size)
        di = dissimilarity_index(self.population, self.facilities, allocation, grp_composition)

        return {
            'alloc_by_facility': alloc_by_facility,
            'facility_rank_distr': facility_rank_distr,
            'capacity': capacity,
            'grp_composition': grp_composition,
            'grp_composition_pct': grp_composition_pct,
            'dissimilarity_index': di
        }