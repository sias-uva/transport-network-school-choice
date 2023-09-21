#%%
import math

import matplotlib
from runner import Runner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from network import Network
%matplotlib inline

plt.rcParams.update({'font.size': 22})
TITLE_FONT_SIZE = 28
SUBTITLE_FONT_SIZE = 26
LEGEND_FONT_SIZE = 16

# env = './envs/amsterdam_neighborhoods/'
env = './envs/grid/GRID_5x10_0.5_[0.8]_lattice'
# env = './envs/sbm/SBM_6_6_0.7_0.01_pop_1000_maj_pop_pct_0.6_0.8_0.9'
# env = './envs/sbm/SBM_6_6_0.3_0.3_pop_1000_maj_pop_pct_0.8'
# env = './envs/sbm/SBM_2_6_0.7_0.01_pop_500_0.5_[0.8]'
# env = './envs/sbm_communities/SBMC_2_25_0.7_[0.8]'
facilities_file = 'facilities.csv'
population_file = 'population.csv'
# network_file = 'network_disconnected_disadvantage.gml'
network_file = 'network.gml'

preferences_model = 'distance_composition'
allocation_model = 'random_serial_dictatorship'
intervention_model = 'none'
simulation_rounds = 30
intervention_rounds = 10
intervention_budget = 5
allocation_rounds = 5
M = 1

preference_model_params = {
    'M': M,
    'init_facility_composition': 'node',
}
update_preference_params = True

c_weights = np.arange(0, 1.1, 0.2)
optimal_group_fractions = np.arange(0, 1.1, 0.2)

#%%
dissimilarity_index = []

for c_weight in np.arange(0, 1, 0.1):
    for optimal_group_fraction in np.arange(0, 1, 0.1):
        print(f'c_weight={c_weight}, optimal_group_fraction={optimal_group_fraction}')

        network = Network(f'{env}/{network_file}', calc_tt_mx=True)
        population = pd.read_csv(f'{env}/{population_file}')
        facilities = pd.read_csv(f'{env}/{facilities_file}')
        runner = Runner(network, population, facilities, logger=None)

        preference_model_params['pop_optimal_grp_frac'] = float(optimal_group_fraction)
        preference_model_params['c_weight'] = float(c_weight)

        di, rwi = runner.run_simulation(
                simulation_rounds,
                allocation_rounds,
                intervention_rounds,
                intervention_budget,
                preferences_model,
                allocation_model, 
                intervention_model, 
                preference_model_params=preference_model_params,
                update_preference_params=update_preference_params)
        
        dissimilarity_index.append([c_weight, optimal_group_fraction, di[-1].mean()])

dissimilarity_index = np.array(dissimilarity_index)

# %% Line chart of the above chart
fig, ax = plt.subplots(figsize = (5, 5))
markers = ['o', 's', 'D', '^', 'v', '*', 'P', 'X', '<', '>']
for i, c_w in enumerate(np.arange(0, 1, 0.1)):
    values = dissimilarity_index[dissimilarity_index[:, 0] == c_w]
    ax.plot(values[:, 1], values[:, 2], marker=markers[i], label=f'alpha: {round(c_w, 1)}', color=matplotlib.cm.get_cmap('plasma_r')(i/10))

ax.set_xlabel('optimal group fraction')
ax.set_ylabel('DI')
ax.set_ylim([0, 1.1])
# fig.suptitle(f'Amsterdam | Segregation for Preference Parameters', fontsize=20, x=0.7)
# fig.suptitle(env + f'{network_file}', fontsize=20)
fig.legend(loc='right', bbox_to_anchor=(1.5, .5), fontsize=18)
fig.show()

#####

# %% Plot different DI lines for different values of the optimal group fraction parameter

def di_progress_by_param(env, pref_model, alloc_model, inter_model, sim_rounds, inter_rounds, inter_budget, alloc_rounds, 
                              c_weights=None, opt_group_frac=None):
    if c_weights is None:
        c_weights = np.round(np.arange(0.1, 1, 0.1), 1)
    if opt_group_frac is None:
        opt_group_frac = np.round(np.arange(0.1, 1, 0.1), 1)

    dissimilarity_index = []
    rounds_with_intervention = []

    for c_weight in c_weights:
        for optimal_group_fraction in opt_group_frac:
            print(f'c_weight={c_weight}, optimal_group_fraction={optimal_group_fraction}')

            network = Network(f'{env}/{network_file}', calc_tt_mx=True)
            population = pd.read_csv(f'{env}/{population_file}')
            facilities = pd.read_csv(f'{env}/{facilities_file}')
            runner = Runner(network, population, facilities, logger=None)

            preference_model_params = {
                'M': M,
                'init_facility_composition': 'node',
            }

            preference_model_params['pop_optimal_grp_frac'] = float(optimal_group_fraction)
            preference_model_params['c_weight'] = float(c_weight)

            di, rwi = runner.run_simulation(
                sim_rounds,
                alloc_rounds,
                inter_rounds,
                inter_budget,
                pref_model,
                alloc_model, 
                inter_model, 
                preference_model_params=preference_model_params,
                update_preference_params=True)
        
            dissimilarity_index.append([c_weight, optimal_group_fraction, di.mean(axis=1)])
            rounds_with_intervention.append([c_weight, optimal_group_fraction, rwi])
    
    dissimilarity_index = np.array(dissimilarity_index)
    rounds_with_intervention = np.array(rounds_with_intervention)

    return dissimilarity_index, rounds_with_intervention

def plot_di_progress_by_param(di, param1, param2, env, pref_model, alloc_model, inter_model, sim_rounds, inter_rounds, inter_budget, alloc_rounds, M, rounds_with_intervention, param1_name='alpha', param2_name='ogf', colors=None, line_styles=None, figsize=(20, 10), ncols=3):
    fig, axs = plt.subplots(math.ceil(len(param1) / ncols), ncols, figsize = figsize)
    
    for i, p1 in enumerate(param1):
        for j, p2 in enumerate(param2):
            ax_i = i // ncols
            ax_j = i % ncols
            values = di[(di[:, 0] == p1) & (di[:, 1] == p2)]
            if colors is not None:
                color = colors[p2]
            else:
                color = None
          
            if line_styles is not None:
                line_style = line_styles[p2]
            else: 
                line_style = None

            axs[ax_i][ax_j].plot(range(len(values[:, 2][0])), values[:, 2][0],
                                label=f'{param2_name}: {p2}',
                                color=color,
                                linestyle=line_style)
            
            axs[ax_i][ax_j].set_ylim([0, 1.1])
            axs[ax_i][ax_j].set_title(f'{param1_name}: {round(p1, 1)}')
            axs[ax_i][ax_j].set_xlabel('round')
            axs[ax_i][ax_j].set_ylabel('DI')

            rwi = rounds_with_intervention[(rounds_with_intervention[:, 0] == p1) & (rounds_with_intervention[:, 1] == p2)]
            for r in rwi[0][2]:
                axs[ax_i][ax_j].axvline(r, linestyle='dashed', linewidth=1, color='#C2C2C2', alpha=0.1)

    handles, labels = axs[0][0].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))

    fig.suptitle(f'{env}/{network_file} \n {pref_model} | {alloc_model} | {inter_model} \n {sim_rounds} simulation rounds, {inter_rounds} intervention rounds, {inter_budget} intervention budget, {alloc_rounds} allocation rounds, M={M}')
    fig.tight_layout()
    fig.show()

def gen_and_plot(env, pref_model, alloc_model, inter_model, sim_rounds, inter_rounds, inter_budget, alloc_rounds, M,
                              c_weights=None, opt_group_frac=None):
    di, rounds_with_intervention = di_progress_by_param(env, 
                              pref_model, 
                              alloc_model, 
                              inter_model, 
                              sim_rounds, 
                              inter_rounds, 
                              inter_budget, 
                              alloc_rounds, 
                              c_weights=c_weights, opt_group_frac=opt_group_frac)
    
    plot_di_progress_by_param(di, 
                            c_weights, 
                            opt_group_frac, 
                            env, 
                            pref_model, 
                            alloc_model, 
                            inter_model,
                            sim_rounds,
                            inter_rounds,
                            inter_budget,
                            alloc_rounds,
                            M,
                            rounds_with_intervention)
    
    return di, rounds_with_intervention

#%%

# di, rwi = gen_and_plot(env=env,
#             pref_model=preferences_model,
#             alloc_model=allocation_model,
#             inter_model='closeness',
#             sim_rounds=simulation_rounds,
#             inter_rounds=intervention_rounds,
#             inter_budget=intervention_budget,
#             alloc_rounds=allocation_rounds,
#             M=M,
#             c_weights=np.round(np.arange(0.1, 1, 0.1), 1),
#             opt_group_frac=np.round(np.arange(0.1, 1, 0.1), 1))


#%% Create a grid plot of all interventions models under different C_weight parameters

def di_progress_by_inter_model(env, pref_model, alloc_model, inter_models: list, sim_rounds, inter_rounds, inter_budget, alloc_rounds, 
                              c_weights=None):
    if c_weights is None:
        c_weights = np.arange(0.1, 1, 0.1)
    # if opt_group_frac is None:
        # opt_group_frac = np.arange(0.1, 1, 0.1)

    dissimilarity_index = []
    rounds_with_intervention = []

    for c_weight in c_weights:
        for inter_model in inter_models:
            print(f'c_weight={c_weight}, inter_model={inter_model}')

            network = Network(f'{env}/{network_file}', calc_tt_mx=True)
            population = pd.read_csv(f'{env}/{population_file}')
            facilities = pd.read_csv(f'{env}/{facilities_file}')
            runner = Runner(network, population, facilities, logger=None)

            preference_model_params = {
                'M': M,
                'init_facility_composition': 'node',
                'pop_optimal_grp_frac': None
            }

            preference_model_params['c_weight'] = float(c_weight)

            di, rwi = runner.run_simulation(
                sim_rounds,
                alloc_rounds,
                inter_rounds,
                inter_budget,
                pref_model,
                alloc_model, 
                inter_model, 
                preference_model_params=preference_model_params,
                update_preference_params=True)
        
            dissimilarity_index.append([c_weight, inter_model, di.mean(axis=1)])
            rounds_with_intervention.append([c_weight, inter_model, rwi])
    
    dissimilarity_index = np.array(dissimilarity_index)
    rounds_with_intervention = np.array(rounds_with_intervention)

    return dissimilarity_index, rounds_with_intervention

inter_models = ['none', 'random', 'closeness', 'group_closeness', 'betweenness', 'group_betweenness']

colors = {
    'none': 'gray', 'random': 'gray', 'closeness': '#e60049', 'group_closeness': '#e60049', 'betweenness': '#0bb4ff', 'group_betweenness': '#0bb4ff', 'degree': '#50e991', 'group_degree': '#50e991'
}
line_styles = {
    'none': '-.', 'random': '--', 'closeness': '-', 'group_closeness': '--', 'betweenness': '-', 'group_betweenness': '--', 'degree': '-', 'group_degree': '--'
}

di, rounds_with_intervention = di_progress_by_inter_model(
            env=env,
            pref_model=preferences_model,
            alloc_model=allocation_model,
            inter_models=inter_models,
            sim_rounds=simulation_rounds,
            inter_rounds=intervention_rounds,
            inter_budget=intervention_budget,
            alloc_rounds=allocation_rounds,
            c_weights=c_weights)

#%%
plot_di_progress_by_param(
            di, 
            c_weights, 
            inter_models,
            env=env,
            pref_model=preferences_model, 
            alloc_model=allocation_model, 
            inter_model=None,
            sim_rounds=simulation_rounds,
            inter_rounds=intervention_rounds,
            inter_budget=intervention_budget,
            alloc_rounds=allocation_rounds,
            M=M,
            rounds_with_intervention=rounds_with_intervention,
            param1_name='alpha',
            param2_name='inter_model',
            colors=colors, 
            line_styles=line_styles,
            figsize=(20, 10),
            ncols=3)

# %%
