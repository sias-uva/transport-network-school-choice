import time
import numpy as np
from logger import Logger
import pandas as pd
from allocation import first_choice, random_serial_dictatorship
from evaluation import calculate_ci, dissimilarity_index, facility_capacity, facility_group_composition, facility_rank_distribution, preference_of_allocation, travel_time_to_allocation
from intervention import create_random_edge, maximize_node_centrality
import matplotlib
import seaborn as sns

from plot import get_figure, heatmap_from_numpy
# Matplotlib stopped working on my machine, so I had to add this line to make it work again.
matplotlib.use("TKAgg")
from network import Network
from preference import distance_composition, distance_popularity, toy_model, nearest_k

class Runner(object):
    def __init__(self, network: Network, population: pd.DataFrame, facilities: pd.DataFrame, logger: Logger):

        self.network = network
        self.population = population
        self.facilities = facilities
        self.logger = logger

        self.facilities_size = facilities.shape[0]
        self.population_size = population.shape[0]
        self.population['group_id'] = population.groupby('group').ngroup()
        self.group_sizes = self.population['group_id'].value_counts()
        self.group_names = self.population.groupby('group_id')['group'].first()
        # For each group, we want to know how its population is distributed over the network nodes. This helps us to calculate metrics weighted by group.
        # We reindex with the node indices to make sure that we have a value for each node, even if the group is not present in that node.
        self.group_node_distr = [(self.population[self.population['group_id'] == gid].groupby('node')['id'].count() / self.group_sizes[gid]).reindex(network.network.vs.indices, fill_value=0) for gid in self.group_names.index]
        self.total_groups = self.population['group_id'].nunique()
        #  Dataframe with node attributes, such as population size and group composition.
        self.nodes = pd.DataFrame(self.network.network.vs.indices, columns=['node'])
        self.nodes = self.nodes.merge(self.population.groupby(['node'])['id'].aggregate(population = 'count'), on='node', how='left')
        # Attach group population and composition to nodes.
        for gid in range(self.total_groups):
            self.nodes = self.nodes.merge(self.population[self.population['group_id'] == gid].groupby('node')['id'].agg(g_pop='count').rename({'g_pop': f'pop_{self.group_names[gid]}'}, axis=1), on='node', how='left').fillna(0)
            self.nodes[f'comp_{self.group_names[gid]}'] = (self.nodes[f'pop_{self.group_names[gid]}'] / self.nodes['population']).fillna(0)
        # Attach relevant node attributes to facilities / keep relevant columns using regex.
        # Group composition of a facility in the beginning is the same as the group composition of the node it is located on.
        self.facilities = self.facilities.merge(self.nodes[self.nodes.filter(regex='node|comp').columns], on='node')
        # Log stuff
        if self.logger:
            logger.append_to_output_file(f'facilities_size: {self.facilities_size}\npopulation_size: {self.population_size}\ntotal_groups: {self.total_groups}')
            for g in self.group_names.index:
                logger.append_to_output_file(f"Group {self.group_names[g]} size: {self.population[self.population['group_id'] == g].shape[0]}")

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

    def run_simulation(self, simulation_rounds: int, allocation_rounds: int, intervention_rounds: int, intervention_budget: int, preferences_model: str, allocation_model: str, intervention_model: str, preference_model_params=None, update_preference_params=False):
        """Runs a simulation of specified simulation_rounds using specified preferences, allocation and intervention models.

        Args:
            simulation_rounds (int): total nr of simulation rounds to run.
            allocation_rounds (int): nr of preference-allocation rounds to run per simulation round.
            intervention_rounds (int): nr of intervention rounds to run per simulation round.
            intervention_budget (int): nr of interventions to do in the network during a single intervention round.
            preferences_model (str): preference model to use.
            allocation_model (str): allocation model to use.
            intervention_model (str): network intervention model to use.
            preference_model_params (dict, optional): controls hyperparameters of the preference model. Defaults to None.
            update_preference_params (bool, optional): whether to update the preference model parameters after each simulation round. Defaults to False.
        """

        # TODO - maybe replace with scenario builder.
        # Note: this currently only runs properly for 2 groups.

        # Note: first round is vanilla - no interventions are added.
        # initialize empty numpy arrays meant to store values of evaluation metrics per simulation round.
        alloc_by_facility = np.zeros((simulation_rounds, allocation_rounds, self.facilities_size))
        capacity = np.zeros((simulation_rounds, allocation_rounds, self.facilities_size))
        popularity = np.zeros((simulation_rounds, self.facilities_size))
        grp_composition_pct = np.zeros((simulation_rounds, allocation_rounds, self.facilities_size, self.total_groups))
        grp_composition = np.zeros((simulation_rounds, allocation_rounds, self.facilities_size, self.total_groups))
        dissimilarity_index = np.zeros((simulation_rounds, allocation_rounds))
        avg_pos_by_fac = np.zeros((simulation_rounds, allocation_rounds, self.facilities_size))
        # Mean travel time to facility for each agent and for each group.
        mean_tt_to_alloc = np.zeros((simulation_rounds, allocation_rounds))
        mean_tt_to_alloc_by_grp = np.zeros((simulation_rounds, allocation_rounds, self.total_groups))
        # Mean utility for each agent and for each group on the assigned facility.
        mean_agent_utility = np.zeros((simulation_rounds, allocation_rounds, self.population_size))
        # Mean position in preferences for allocated facilities for each agent and for each group.
        mean_pos_of_alloc = np.zeros((simulation_rounds, allocation_rounds))
        mean_pos_of_alloc_by_grp = np.zeros((simulation_rounds, allocation_rounds, self.total_groups))
        interventions = []
        # All the rounds where an intervention happened.
        rounds_with_intervention = []

        for i in range(simulation_rounds):
            # On the first round, we don't want to add any interventions, just run an agent round.
            # After that, we want to have a total of intervention_rounds evenly spread in the simulations.
            intervention_round = intervention_rounds > 0 and i > 0 and ((i == 1) or i % (simulation_rounds // intervention_rounds) == 0)
            if intervention_round:
                intervention = self.create_intervention(intervention_model)
                interventions.append(intervention)
                rounds_with_intervention.append(i)

            # Store all preference lists for each allocation round for each agent.
            # This might break if preferences return less items than the total facility size.
            pref_lists = np.zeros((allocation_rounds, self.population_size, self.facilities_size))
            # Store all allocation lists for each allocation round for each agent.
            alloc_lists = np.zeros((allocation_rounds, self.population_size, self.facilities_size))
            for j in range(allocation_rounds):
                pref_list, utility, allocation, eval_metrics = self.run_agent_round(preferences_model, allocation_model, preference_model_params)
                pref_lists[j] = pref_list
                alloc_lists[j] = allocation
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
                mean_agent_utility[i][j] = utility[range(self.population_size), allocation.flatten()]

            if self.logger:
                # Makes no sense to plot this for a lot of facilities
                # alloc_heatmap = heatmap_from_numpy(grp_composition.mean(axis=1)[i], 
                #                         title=f"Allocation by facility and group (mean) - round {i}", 
                #                         subtitle=f"{preferences_model} - {allocation_model} - {intervention_model}",
                #                         figsize=(10, 10),
                #                         xlabel='Groups',
                #                         ylabel='Facilities')
                # self.logger.save_plot(alloc_heatmap, f"allocation_by_facility_and_group_{i}.png", round=i)

                # Makes no sense to plot this for a lot of facilities
                # rank_distribution_heatmap = heatmap_from_numpy(eval_metrics['facility_rank_distr'], 
                #                             title=f"Facility rank distribution - round {i}",
                #                             subtitle=f"{preferences_model} - {allocation_model} - {intervention_model}",
                #                             xlabel='Rank',
                #                             ylabel='Facilities')
                # self.logger.save_plot(rank_distribution_heatmap, f"rank_distribution_{i}.png", round=i)

                # Create a matplotlib plot with the distribution of utility for each agent, based on the assigned facility.
                # TODO NOTE: This doubles the running time of the simulation, -- I am thinking maybe we can store everything and then plot it all at the end?
                fig, ax = get_figure(f"Utility distribution for agents on assigned facility - round {i}",
                                f"{preferences_model} - {allocation_model} - {intervention_model}",
                                xlabel='Utility',
                                ylabel='Frequency')
                sns.histplot(mean_agent_utility[i].flatten(), stat='probability', bins=10)
                ax.axvline(np.median(mean_agent_utility[i].flatten()), linestyle='dashed', linewidth=1)
                self.logger.save_plot(fig, f"agent_utility_distribution_{i}.png", round=i)

                fig, ax = get_figure(f"Utility distribution for groups on assigned facility - round {i}",
                                f"{preferences_model} - {allocation_model} - {intervention_model}",
                                xlabel='Utility',
                                ylabel='Frequency')
                
                group_memberships = self.population['group_id'].values
                sns.histplot([mean_agent_utility[i, :, group_memberships == g_id].flatten() for g_id in self.group_names.index], 
                                stat='probability',
                                bins=10, 
                                ax=ax) 
                
                for g_id in self.group_names.index:
                    ax.axvline(np.median(mean_agent_utility[i, :, group_memberships == g_id].flatten()), color=f"C{g_id}", linestyle='dashed', linewidth=1)
                
                ax.legend(labels=[f"Group {self.group_names[g_id]}" for g_id in self.group_names.index])
                self.logger.save_plot(fig, f"group_utility_distribution_{i}.png", round=i)

                # Filtered out cause its useless when the network is large and increases runtime 10x.
                # travel_time_heatmap = heatmap_from_numpy(self.network.tt_mx,
                #                             title=f"Travel Time between Nodes - round {i}",
                #                             subtitle=f"{preferences_model} - {allocation_model} - {intervention_model}",
                #                             xlabel='Nodes',
                #                             ylabel='Nodes')
                # self.logger.save_plot(travel_time_heatmap, f"travel_time_matrix{i}.png", round=i)

                # Create a plot with the network intervention.
                if intervention_round:
                    self.logger.save_igraph_plot(self.network, f"intervention_{i}.pdf", edges_to_color=intervention , round=i)
            
            if update_preference_params:
                # Keep a record of the popularity of each facility for each round.
                popularity[i] = self.facilities['popularity'].values
                # Update the preference parameters for the next round.
                self.update_preference_parameters(pref_lists, grp_composition_pct[i])

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
                    ax.plot(range(simulation_rounds), grpcomp_ci[:, 0, fid, gid], label=f'Group {self.group_names[gid]}')
                    ax.fill_between(range(simulation_rounds), grpcomp_ci[:, 1, fid, gid], grpcomp_ci[:, 2, fid, gid], alpha=.2)
                ax.hlines(y=0.5, xmin=0, xmax=simulation_rounds-1, color='gray', linestyle='--')
                ax.legend()

                self.logger.save_plot(fig, f'facility_{fid}_group_composition.png')

            # Generate Dissimilarity Index plot for all facilities.
            diss_ci = np.apply_along_axis(calculate_ci, 1, dissimilarity_index)
            self.logger.log_numpy_array(diss_ci, 'dissimilarity_index.txt')
            fig, ax = get_figure(f"Dissimilarity Index",
                                f"{preferences_model} - {allocation_model} - {intervention_model}",
                                xlabel='Simulation round',
                                ylabel='Dissimilarity Index',
                                ylim=(0, 1))
            ax.plot(range(simulation_rounds), diss_ci[:, 0], label=f'Dissimilarity Index')
            ax.fill_between(range(simulation_rounds), diss_ci[:, 1], diss_ci[:, 2], color='b', alpha=.1)
            for r in rounds_with_intervention:
                ax.axvline(r, linestyle='dashed', linewidth=1, color='gray')
            
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

            # Generate Popularity plot for all facilities.
            fig, ax = get_figure(f"Facility Popularity",
                                f"{preferences_model} - {allocation_model} - {intervention_model}",
                                xlabel='Simulation round',
                                ylabel='Popularity')
            for fid in range(self.facilities_size):
                ax.plot(range(simulation_rounds), popularity[:, fid], label=f'Facility {fid}')
            ax.legend()
            self.logger.save_plot(fig, f'facility_popularity.png')

            # Generate Mean Travel Time to Allocation plot.
            mttalloc_ci = np.apply_along_axis(calculate_ci, 1, mean_tt_to_alloc)
            mttallocgrp_ci = np.apply_along_axis(calculate_ci, 1, mean_tt_to_alloc_by_grp)
            fig, ax = get_figure(f"Mean Travel Time to Allocated Facility",
                                f"{preferences_model} - {allocation_model} - {intervention_model}",
                                xlabel='Simulation round',
                                ylabel='Mean Travel Time')
            ax.plot(range(simulation_rounds), mttalloc_ci[:, 0], label=f'Mean Travel Time', color='#C4C4C4')
            ax.fill_between(range(simulation_rounds), mttalloc_ci[:, 1], mttalloc_ci[:, 2], alpha=.1)
            [ax.plot(range(simulation_rounds), mttallocgrp_ci[:, 0, g], label=f'Group {self.group_names[g]}') for g in range(self.total_groups)]
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
            [ax.plot(range(simulation_rounds), mposallocgrp_ci[:, 0, g], label=f'Group {self.group_names[g]}', color=f'C{g}') for g in range(self.total_groups)]
            [ax.fill_between(range(simulation_rounds), mposallocgrp_ci[:, 1, g], mposallocgrp_ci[:, 2, g], alpha=.1, color=f'C{g}') for g in range(self.total_groups)]
            fig.legend()

            # Save the mean position in pref list plot.
            self.logger.save_plot(fig, f'mean_pref_position_of_allocation.png')

            # Generate plot of all network interventions
            self.logger.save_igraph_plot(self.network, f"network_interventions.pdf", edges_to_color=self.network.added_edges)
            # Save all network interventions and their rounds to the output file.
            self.logger.append_to_output_file(f"interventions: {[(i.source, i.target) for i in interventions if i is not None]}")
            self.logger.append_to_output_file(f"rounds_with_intervention: {rounds_with_intervention}")
        
    def run_agent_round(self, preferences_model, allocation_model, preference_model_params=None):
        """Runs a round of preference generation -> allocation generation -> evaluation.

        Args:
            preferences_model (str): model to use to generate preferences
            allocation_model (str): model to use to generate allocations
            preference_model_params (dict, optional): controls hyperparameters of the preference model. Defaults to None.

        Returns:
            list: preference_list, allocation, capacity_eval, diversity_eval
        """
        pref_list, utility = self.generate_preferences(preferences_model, preference_model_params=preference_model_params, return_utility=True)
        allocation = self.generate_allocation(pref_list, allocation_model)
        eval_metrics = self.evaluate(pref_list, allocation)

        return pref_list, utility, allocation, eval_metrics

    def generate_preferences(self, preferences_model: str, preference_model_params=None, return_utility=False):
        """Generates preferences for each agent in the population, according to preferences_model.

        Args:
            preferences_model (str): preference model to use.
            preference_model_params (dict, optional): controls hyperparameters of the preference model. Defaults to None.
            return_utility (bool, optional): whether to return the utility of each agent (the score assigned to each facility). Defaults to False.
        Returns:
            - np.array: array of size (nr of agents, nr of facilities) where each facility is sorted by preference.
            - np.array: array of size (nr of agents, nr of facilities) where each facility is assigned a utility score.
        """
        pref_list = None
        utility = None
        travel_time = self.network.tt_mx[self.population['node'].values][:, [self.facilities['node'].values]].squeeze()
        if preferences_model == 'nearest_k':
            assert 'nearest_k' in preference_model_params.keys(), 'You need to specify nearest_k parameter in config.'
            pref_list, utility = nearest_k(travel_time, k=preference_model_params['nearest_k'])
        elif preferences_model == 'toy_model':
            assert 'quality' in self.facilities.columns, 'To use the toy_model preference model, the facilities_file should contain a column named "quality".'
            # Select facility qualities
            qualities = self.facilities['quality'].to_numpy()
            pref_list, utility = toy_model(travel_time, qualities)
        elif preferences_model == 'distance_popularity':
            assert 'popularity' in self.facilities.columns, 'To use the distance_popularity preference model, the facilities_file should contain a column named "popularity".'
            # Select facility popularities
            popularity = self.facilities['popularity'].to_numpy()
            pref_list, utility = distance_popularity(travel_time, popularity)
        elif preferences_model == 'distance_composition':
            assert 'tolerance' in self.population.columns, 'To use the distance_composition preference model, the population of agents should contain a column named "tolerance".'
            # Select facility compositions
            pref_list, utility = distance_composition(travel_time, self.population, self.facilities, M=preference_model_params['M'], C_weight=preference_model_params['c_weight'])

        assert pref_list is not None, 'No preference list was generated, specify a valid preferences_model parameter in config.'
        
        if return_utility:
            return pref_list, utility
        else: 
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
        fac_nodes = self.facilities['node'].values
        if intervention_model == 'none':
            return
        elif intervention_model == 'random':
            x, y, w = create_random_edge(self.network)
        elif intervention_model == 'closeness':
            # Find the facility with the lowest closeness centrality to augment, then find the edge that maximizes that node's centrality.
            node_to_augment = fac_nodes[np.argmin(self.network.network.closeness(fac_nodes))].item()
            x, y, w = maximize_node_centrality(self.network, node_to_augment, 'closeness')
        elif intervention_model == 'betweenness':
            # Find the facility with the lowest betweenness centrality to augment, then find the edge that maximizes that node's centrality.
            node_to_augment = fac_nodes[np.argmin(self.network.network.betweenness(fac_nodes))].item()
            x, y, w = maximize_node_centrality(self.network, node_to_augment, 'betweenness')
        elif intervention_model == 'degree':
            # Find the facility with the lowest degree centrality to augment, then find the edge that maximizes that node's centrality.
            node_to_augment = fac_nodes[np.argmin(self.network.network.degree(fac_nodes))].item()
            x, y, w = maximize_node_centrality(self.network, node_to_augment, 'degree')
        elif intervention_model == 'group_closeness':
            group_closenesses = np.array([self.network.weighted_closeness(fac_nodes, weights=self.group_node_distr[gid].values) for gid in self.group_names.index])
            # Returns a tuple of (group_id, node_id) where node_id is the node with the lowest closeness with respect to group_id.
            grp_to_augment, node_idx_to_augment = np.unravel_index(group_closenesses.argmin(), group_closenesses.shape)
            # node_idx_to_augment is the index of the node in the group_node_distr array, we need to get the actual node id.
            node_to_augment = fac_nodes[node_idx_to_augment].item()
            x, y, w = maximize_node_centrality(self.network, node_to_augment, 'group_closeness', group_weights=self.group_node_distr[grp_to_augment].values)
        elif intervention_model == 'group_betweenness':
            group_betweenness = np.array([self.network.weighted_betweeness(fac_nodes, weights=self.group_node_distr[gid].values) for gid in self.group_names.index])
            # Returns a tuple of (group_id, node_id) where node_id is the node with the lowest betweenness with respect to group_id.
            grp_to_augment, node_idx_to_augment = np.unravel_index(group_betweenness.argmin(), group_betweenness.shape)
            # node_idx_to_augment is the index of the node in the group_node_distr array, we need to get the actual node id.
            node_to_augment = fac_nodes[node_idx_to_augment].item()
            x, y, w = maximize_node_centrality(self.network, node_to_augment, 'group_betweenness', group_weights=self.group_node_distr[grp_to_augment].values)
        elif intervention_model == 'group_degree':
            group_degree = np.array([self.network.weighted_degree(fac_nodes, weights=self.group_node_distr[gid].values) for gid in self.group_names.index])
            # Returns a tuple of (group_id, node_id) where node_id is the node with the lowest degree with respect to group_id.
            grp_to_augment, node_idx_to_augment = np.unravel_index(group_degree.argmin(), group_degree.shape)
            # node_idx_to_augment is the index of the node in the group_node_distr array, we need to get the actual node id.
            node_to_augment = fac_nodes[node_idx_to_augment].item()
            x, y, w = maximize_node_centrality(self.network, node_to_augment, 'group_degree', group_weights=self.group_node_distr[grp_to_augment].values)
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
        mean_tt_to_alloc, mean_tt_to_alloc_by_group = travel_time_to_allocation(self.network.tt_mx, self.population, self.facilities, allocation, return_group_avg=True, groups=self.group_names)
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

    def update_preference_parameters(self, pref_lists, grp_composition_pct):
        """
        Updates parameters related to the preference models, such as popularity, group composition, etc. It should only run if dynamic_preference_model is set to True.

        Args:
            pref_lists (np.array): array of size (allocation_rounds, nr_agents, nr_facilities) where each agent has a list of preferences.
            grp_composition_pct (np.array): array of size (allocation_rounds, nr_facilities, nr_groups) where each facility has a group composition.

        Returns:
            None
        """

        # 1. Add +1 to positions to avoid division by zero.
        # 2. Get the reciprical of the positions, so that the first choice has the highest weight.
        # 3. Calculate a weighted avg of the preferences for each facility, set this as the new popularity.
        popularity = [np.mean(1/(np.where(pref_lists == f)[2] + 1)) for f in self.facilities['id'].values]
        self.facilities['popularity'] = popularity

        # Average over all the allocation rounds to get the average group composition per facility.
        # Assign the new group composition to the facilities.
        self.facilities[[f'comp_g{g}' for g in range(grp_composition_pct.shape[1])]] = grp_composition_pct.mean(axis=0)
