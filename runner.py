import time
import numpy as np
from logger import Logger
import pandas as pd
from allocation import first_choice, random_serial_dictatorship
from evaluation import calculate_ci, dissimilarity_index, facility_capacity, facility_group_composition, facility_rank_distribution, preference_of_allocation, travel_time_to_allocation
from intervention import create_random_edge, maximize_closeness_centrality
import matplotlib

from plot import get_figure, heatmap_from_numpy
# Matplotlib stopped working on my machine, so I had to add this line to make it work again.
matplotlib.use("TKAgg")
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
        self.population['group_id'] = population['group'].apply(lambda x: np.searchsorted(self.groups, x))

        self.total_groups = population['group'].nunique()
        # Log stuff
        if self.logger:
            logger.append_to_output_file(f'facilities_size: {self.facilities_size}\npopulation_size: {self.population_size}\ntotal_groups: {self.total_groups}')
            for g in self.groups:
                logger.append_to_output_file(f"Group {g} size: {population[population['group'] == g].shape[0]}")

        # Calculate travel times for all agents in the population to all facilities.
        travel_time = self.network.tt_mx[self.population['node'].values][:, [self.facilities['node'].values]].squeeze()

        ## Just a random check if the above indexing works, to remove later on.
        pop_sample = population.sample()
        fac_sample = facilities.sample()
        assert self.network.tt_mx[pop_sample.iloc[0]['node'], fac_sample.iloc[0]['node']] == travel_time[pop_sample.iloc[0]['id'], fac_sample.iloc[0]['id']], "Something wrong with travel time indexing - incompatible travel times between network pre-calculated and indexed values."
        ##

        # Safe a figure of the network to the output folder.
        if self.logger:
            self.logger.save_igraph_plot(self.network, facilities_to_label=self.facilities['node'].values)

    def run_simulation(self, simulation_rounds: int, allocation_rounds: int, preferences_model: str, allocation_model: str, intervention_model: str, nearest_k_k=None):
        """Runs a simulation of specified simulation_rounds using specified preferences, allocation and intervention models.

        Args:
            simulation_rounds (int): total nr of simulation rounds to run.
            allocation_rounds (int): nr of preference-allocation rounds to run per simulation round.
            preferences_model (str): preference model to use.
            allocation_model (str): allocation model to use.
            intervention_model (str): network intervention model to use.
            nearest_k_k (int, optional): k parameter in nearest_k preference model. Defaults to None.
        """

        # TODO - maybe replace with scenario builder.
        # Note: this currently only runs properly for 2 groups.

        # Note: first round is vanilla - no interventions are added.
        # _, _, eval_metrics = self.run_agent_round(preferences_model, allocation_model, nearest_k_k)
        # initialize empty numpy arrays meant to store values of evaluation metrics per simulation round.
        alloc_by_facility = np.zeros((simulation_rounds, allocation_rounds, self.facilities_size))
        capacity = np.zeros((simulation_rounds, allocation_rounds, self.facilities_size))
        grp_composition_pct = np.zeros((simulation_rounds, allocation_rounds, self.facilities_size, self.total_groups))
        grp_composition = np.zeros((simulation_rounds, allocation_rounds, self.facilities_size, self.total_groups))
        dissimilarity_index = np.zeros((simulation_rounds, allocation_rounds))
        avg_pos_by_fac = np.zeros((simulation_rounds, allocation_rounds, self.facilities_size))
        # Mean travel time to facility for each agent and for each group.
        mean_tt_to_alloc = np.zeros((simulation_rounds, allocation_rounds))
        mean_tt_to_alloc_by_grp = np.zeros((simulation_rounds, allocation_rounds, self.total_groups))
        # Mean position in preferences for allocated facilities for each agent and for each group.
        mean_pos_of_alloc = np.zeros((simulation_rounds, allocation_rounds))
        mean_pos_of_alloc_by_grp = np.zeros((simulation_rounds, allocation_rounds, self.total_groups))
        interventions = []

        for i in range(simulation_rounds):
            # On the first round, we don't want to add any interventions, just run an agent round.
            if i > 0:
                intervention = self.create_intervention(intervention_model)
                interventions.append(intervention)

            for j in range(allocation_rounds):
                pref_list, _, eval_metrics = self.run_agent_round(preferences_model, allocation_model, nearest_k_k)
                # Log pref_list to a file.
                if self.logger:
                    agentpref = self.population.copy()
                    agentpref['pref_list'] = pref_list.tolist()
                    self.logger.log_dataframe(agentpref, f'agents_pref_list_{i}.csv', round=i)
                
                alloc_by_facility[i][j] = eval_metrics['alloc_by_facility']
                capacity[i][j] = eval_metrics['capacity']
                grp_composition_pct[i][j] = eval_metrics['grp_composition_pct']
                grp_composition[i][j] = eval_metrics['grp_composition']
                dissimilarity_index[i][j] = eval_metrics['dissimilarity_index']
                avg_pos_by_fac[i][j] = eval_metrics['avg_pos_by_fac']
                mean_tt_to_alloc[i][j] = eval_metrics['mean_tt_to_alloc']
                mean_tt_to_alloc_by_grp[i][j] = eval_metrics['mean_tt_to_alloc_by_group']
                mean_pos_of_alloc[i][j] = eval_metrics['pref_of_alloc']
                mean_pos_of_alloc_by_grp[i][j] = eval_metrics['pref_of_alloc_by_group']

            if self.logger:
                alloc_heatmap = heatmap_from_numpy(grp_composition.mean(axis=1)[i], 
                                        title=f"Allocation by facility and group (mean) - round {i}", 
                                        subtitle=f"{preferences_model} - {allocation_model} - {intervention_model}",
                                        figsize=(10, 10),
                                        xlabel='Groups',
                                        ylabel='Facilities')
                self.logger.save_plot(alloc_heatmap, f"allocation_by_facility_and_group_{i}.png", round=i)

                rank_distribution_heatmap = heatmap_from_numpy(eval_metrics['facility_rank_distr'], 
                                            title=f"Facility rank distribution - round {i}",
                                            subtitle=f"{preferences_model} - {allocation_model} - {intervention_model}",
                                            xlabel='Rank',
                                            ylabel='Facilities')
                self.logger.save_plot(rank_distribution_heatmap, f"rank_distribution_{i}.png", round=i)

                # Filtered out cause its useless when the network is large and increases runtime 10x.
                # travel_time_heatmap = heatmap_from_numpy(self.network.tt_mx,
                #                             title=f"Travel Time between Nodes - round {i}",
                #                             subtitle=f"{preferences_model} - {allocation_model} - {intervention_model}",
                #                             xlabel='Nodes',
                #                             ylabel='Nodes')
                # self.logger.save_plot(travel_time_heatmap, f"travel_time_matrix{i}.png", round=i)

                # Create a plot with the network intervention.
                if i > 0 :
                    self.logger.save_igraph_plot(self.network, f"intervention_{i}.pdf", edges_to_color=intervention , round=i)

        if self.logger:
            # Generate group composition plot for each facility (diffrent plots).
            grpcomp_ci = np.apply_along_axis(calculate_ci, 1, grp_composition_pct)
            for fid in range(self.facilities_size):
                fig, ax = get_figure(f"Facility {self.facilities.iloc[fid]['facility']} ({fid}) - Group Composition",
                                    f"{preferences_model} - {allocation_model} - {intervention_model}",
                                    xlabel='Simulation round',
                                    ylabel='Group composition',
                                    ylim=(0, 1))
                for gid in range(self.total_groups):
                    # plot the mean
                    ax.plot(range(simulation_rounds), grpcomp_ci[:, 0, fid, gid], label=f'Group {gid}')
                    ax.fill_between(range(simulation_rounds), grpcomp_ci[:, 1, fid, gid], grpcomp_ci[:, 2, fid, gid], alpha=.2)
                ax.hlines(y=0.5, xmin=0, xmax=simulation_rounds-1, color='gray', linestyle='--')
                ax.legend()

                self.logger.save_plot(fig, f'facility_{fid}_group_composition.png')

            # Generate Dissimilarity Index plot for all facilities.
            diss_ci = np.apply_along_axis(calculate_ci, 1, dissimilarity_index)
            fig, ax = get_figure(f"Dissimilarity Index",
                                f"{preferences_model} - {allocation_model} - {intervention_model}",
                                xlabel='Simulation round',
                                ylabel='Dissimilarity Index',
                                ylim=(0, 1))
            ax.plot(range(simulation_rounds), diss_ci[:, 0], label=f'Dissimilarity Index')
            ax.fill_between(range(simulation_rounds), diss_ci[:, 1], diss_ci[:, 2], color='b', alpha=.1)
            
            self.logger.save_plot(fig, f'dissimilarity_index.png')

            # Generate Average Preference Position for all facilities.
            fig, ax = get_figure("Average Preference Position", 
                                f"{preferences_model} - {allocation_model} - {intervention_model}", 
                                xlabel="Simulation round", 
                                ylabel="Average Preference Position")
            
            facpref_ci = np.apply_along_axis(calculate_ci, 1, avg_pos_by_fac)
            for fid in range(self.facilities_size):
                ax.plot(range(simulation_rounds), facpref_ci[:, 0, fid], label=f'Facility {fid}')
                ax.fill_between(range(simulation_rounds), facpref_ci[:, 1, fid], facpref_ci[:, 2, fid], color='b', alpha=.1)

            ax.legend()
            self.logger.save_plot(fig, f'average_facility_pref_position.png')
        
            # Generate Capacity plot for all facilities.
            cap_ci = np.apply_along_axis(calculate_ci, 1, capacity)
            fig, ax = get_figure(f"Facility Capacity",
                                f"{preferences_model} - {allocation_model} - {intervention_model}",
                                xlabel='Simulation round',
                                ylabel='Capacity %',
                                ylim=(0, 2))
            for fid in range(self.facilities_size):
                ax.plot(range(simulation_rounds), cap_ci[:, 0, fid], label=f'Facility {fid}')
                ax.fill_between(range(simulation_rounds), cap_ci[:, 1, fid], cap_ci[:, 2, fid], alpha=.1)
            ax.legend()
            self.logger.save_plot(fig, f'facility_capacity.png')

            # Generate Mean Travel Time to Allocation plot.
            mttalloc_ci = np.apply_along_axis(calculate_ci, 1, mean_tt_to_alloc)
            mttallocgrp_ci = np.apply_along_axis(calculate_ci, 1, mean_tt_to_alloc_by_grp)
            fig, ax = get_figure(f"Mean Travel Time to Allocated Facility",
                                f"{preferences_model} - {allocation_model} - {intervention_model}",
                                xlabel='Simulation round',
                                ylabel='Mean Travel Time')
            ax.plot(range(simulation_rounds), mttalloc_ci[:, 0], label=f'Mean Travel Time', color='#C4C4C4')
            ax.fill_between(range(simulation_rounds), mttalloc_ci[:, 1], mttalloc_ci[:, 2], alpha=.1)
            [ax.plot(range(simulation_rounds), mttallocgrp_ci[:, 0, g], label=f'Group {self.groups[g]}') for g in range(self.total_groups)]
            [ax.fill_between(range(simulation_rounds), mttallocgrp_ci[:, 1, g], mttallocgrp_ci[:, 2, g], alpha=.1, color=f'C{g}') for g in range(self.total_groups)]
            fig.legend()
            
            # Save the mean travel time plot.
            self.logger.save_plot(fig, f'mean_tt_to_allocation.png')

            # Generate Mean Position in pref list for allocation plot
            mposalloc_ci = np.apply_along_axis(calculate_ci, 1, mean_pos_of_alloc)
            mposallocgrp_ci = np.apply_along_axis(calculate_ci, 1, mean_pos_of_alloc_by_grp)
            fig, ax = get_figure(f"Mean Position in Preference of Allocated Facility",
                                f"{preferences_model} - {allocation_model} - {intervention_model}",
                                xlabel='Simulation round',
                                ylabel='Mean Position in Preference List')
            ax.plot(range(simulation_rounds), mposalloc_ci[:, 0], label=f'Mean Position', color='#C4C4C4')
            ax.fill_between(range(simulation_rounds), mposalloc_ci[:, 1], mposalloc_ci[:, 2], alpha=.1)
            [ax.plot(range(simulation_rounds), mposallocgrp_ci[:, 0, g], label=f'Group {self.groups[g]}', color=f'C{g}') for g in range(self.total_groups)]
            [ax.fill_between(range(simulation_rounds), mposallocgrp_ci[:, 1, g], mposallocgrp_ci[:, 2, g], alpha=.1, color=f'C{g}') for g in range(self.total_groups)]
            fig.legend()

            # Save the mean position in pref list plot.
            self.logger.save_plot(fig, f'mean_pref_position_of_allocation.png')

            # Generate plot of all network interventions
            self.logger.save_igraph_plot(self.network, f"network_interventions.pdf", edges_to_color=self.network.added_edges)
            # Save all network interventions to the output file.
            self.logger.append_to_output_file(f"interventions: {[(i.source, i.target) for i in interventions if i is not None]}")
        
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
        travel_time = self.network.tt_mx[self.population['node'].values][:, [self.facilities['node'].values]].squeeze()
        if preferences_model == 'nearest_k':
            assert nearest_k_k, 'You need to specify nearest_k parameter in config.'
            pref_list = nearest_k(travel_time, k=nearest_k_k)
        elif preferences_model == 'toy_model':
            # Select facility qualities
            qualities = self.facilities.quality.to_numpy()
            pref_list = toy_model(travel_time, qualities)

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
        elif intervention_model == 'closeness':
            # Find the facility with the lowest closeness centrality to augment, then find the edge that maximizes that node's centrality.
            fac_nodes = self.facilities['node'].values
            node_to_augment = fac_nodes[np.argmin(self.network.network.closeness(fac_nodes))].item()
            x, y, w = maximize_closeness_centrality(self.network, node_to_augment)
        else:
            assert False, 'No intervention was generated, specify a valid intervention_model parameter in config.'

        if x is None:
            print('No intervention was created.')
            return None
        else:
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
        facility_rank_distr, avg_pos_by_fac = facility_rank_distribution(pref_list, self.facilities_size, return_avg_pos_by_fac=True)
        di = dissimilarity_index(self.population, self.facilities, allocation, grp_composition)
        mean_tt_to_alloc, mean_tt_to_alloc_by_group = travel_time_to_allocation(self.network.tt_mx, self.population, self.facilities, allocation, return_group_avg=True, groups=self.groups)
        pref_of_alloc, pref_of_alloc_by_group = preference_of_allocation(pref_list, allocation, return_group_avg=True, group_membership=self.population['group_id'].values)

        return {
            'alloc_by_facility': alloc_by_facility,
            'facility_rank_distr': facility_rank_distr,
            'avg_pos_by_fac': avg_pos_by_fac,
            'capacity': capacity,
            'grp_composition': grp_composition,
            'grp_composition_pct': grp_composition_pct,
            'dissimilarity_index': di,
            'mean_tt_to_alloc': mean_tt_to_alloc,
            'mean_tt_to_alloc_by_group': mean_tt_to_alloc_by_group,
            'pref_of_alloc': pref_of_alloc,
            'pref_of_alloc_by_group': pref_of_alloc_by_group
        }