#%%
import math
from runner import Runner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from network import Network
%matplotlib inline

######### SBM #########
#%% Here we test what happens in a non-segregated environment, when we adapt the optimal group fraction parameter (homophily).

# env = './envs/sbm/SBM_6_6_0.7_0.01_pop_1000_maj_pop_pct_0.5'
# env = './envs/sbm/SBM_6_6_0.7_0.01_pop_1000_maj_pop_pct_0.6_0.8_0.9'

# network = Network(f'{env}/network.gml', calc_tt_mx=True)
# population = pd.read_csv(f'{env}/population.csv')
# facilities = pd.read_csv(f'{env}/facilities.csv')
# preferences_model = 'distance_composition'
# allocation_model = 'random_serial_dictatorship'
# intervention_model = 'none'
# simulation_rounds = 30
# intervention_rounds = 2
# intervention_budget = 1
# allocation_rounds = 5

# preference_model_params = {
#     'c_weight': 0.1,
#     'M': 0.6,
#     'init_facility_composition': 'node',
# }
# update_preference_params = True


# runner = Runner(network, population, facilities, logger=None)

# optimal_group_fractions = np.arange(0.1, 1, 0.1)

# dissimilarity_index = []

# for optimal_group_fraction in optimal_group_fractions:
#     preference_model_params['pop_optimal_grp_frac'] = float(optimal_group_fraction)

#     di = runner.run_simulation(
#             simulation_rounds,
#             allocation_rounds,
#             intervention_rounds,
#             intervention_budget,
#             preferences_model,
#             allocation_model, 
#             intervention_model, 
#             preference_model_params=preference_model_params,
#             update_preference_params=update_preference_params)
    
#     dissimilarity_index.append(di[-1].mean())

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot(optimal_group_fractions, dissimilarity_index, marker='o', label=f'optimal_group_fraction={optimal_group_fraction}')
# ax.set_xlabel('optimal_group_fraction')
# ax.set_ylabel('dissimilarity_index')
# ax.set_ylim([0, 1])
# fig.show()

# #%% Run c_weights sensitivity analysis for the SBM environment

# preference_model_params = {
#     'M': 0.6,
#     'init_facility_composition': 'node',
#     'pop_optimal_grp_frac': 0.5
# }

# runner = Runner(network, population, facilities, logger=None)

# c_weights = np.arange(0.1, 1, 0.1)

# dissimilarity_index = []

# for c_weight in c_weights:
#     preference_model_params['c_weight'] = float(c_weight)

#     di = runner.run_simulation(
#             simulation_rounds,
#             allocation_rounds,
#             intervention_rounds,
#             intervention_budget,
#             preferences_model,
#             allocation_model, 
#             intervention_model, 
#             preference_model_params=preference_model_params,
#             update_preference_params=update_preference_params)
    
#     dissimilarity_index.append(di[-1].mean())

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot(c_weights, dissimilarity_index, marker='o')
# ax.set_xlabel('c_weight')
# ax.set_ylabel('dissimilarity_index')
# ax.set_ylim([0, 1])
# fig.show()

# #%% Run 2d sensitivity analysis for the SBM environment - c_weight, optimal_group_fraction

# preference_model_params = {
#     'M': 0.6,
#     'init_facility_composition': 'node',
# }
# update_preference_params = True


# runner = Runner(network, population, facilities, logger=None)

# c_weights = np.arange(0.1, 1, 0.1)
# optimal_group_fractions = np.arange(0.1, 1, 0.1)

# dissimilarity_index = []

# for c_weight in c_weights:
#     for optimal_group_fraction in optimal_group_fractions:
#         preference_model_params['pop_optimal_grp_frac'] = float(optimal_group_fraction)
#         preference_model_params['c_weight'] = float(c_weight)

#         di = runner.run_simulation(
#                 simulation_rounds,
#                 allocation_rounds,
#                 intervention_rounds,
#                 intervention_budget,
#                 preferences_model,
#                 allocation_model, 
#                 intervention_model, 
#                 preference_model_params=preference_model_params,
#                 update_preference_params=update_preference_params)
        
#         dissimilarity_index.append([c_weight, optimal_group_fraction, di[-1].mean()])

# #%% Create 3d plot for the SBM environment - c_weight, optimal_group_fraction, dissimilarity_index
# dissimilarity_index = np.array(dissimilarity_index)
# fig = plt.figure(figsize = (5, 5))
# ax = plt.axes(projection ="3d")

# # ax.scatter(c_weights, optimal_group_fractions, dissimilarity_index, color = "green", marker = "o")
# ax.scatter(dissimilarity_index[:, 0], dissimilarity_index[:, 1], dissimilarity_index[:, 2], color = "green", marker = "o")

# # Set axis labels
# ax.set_xlabel('C_weight')
# ax.set_ylabel('optimal_group_fraction')
# ax.set_zlabel('dissimilarity_index')
# ax.set_zlim([0, 1])
# fig.show()

# #%%
# ######### AMSTERDAM #########
# env = './envs/amsterdam_neighborhoods'

# network = Network(f'{env}/network.gml', calc_tt_mx=True)
# population = pd.read_csv(f'{env}/population.csv')
# facilities = pd.read_csv(f'{env}/schools.csv')
# preferences_model = 'distance_composition'
# allocation_model = 'random_serial_dictatorship'
# intervention_model = 'none'
# simulation_rounds = 50
# intervention_rounds = 25
# intervention_budget = 1
# allocation_rounds = 5

# #%% Optimal Fraction Parameter
# preference_model_params = {
#     'c_weight': 0.5,
#     'M': 0.6,
#     'init_facility_composition': 'node',
# }
# update_preference_params = True


# runner = Runner(network, population, facilities, logger=None)

# optimal_group_fractions = np.arange(0.1, 1, 0.1)

# dissimilarity_index = []

# for optimal_group_fraction in optimal_group_fractions:
#     preference_model_params['pop_optimal_grp_frac'] = float(optimal_group_fraction)

#     di = runner.run_simulation(
#             simulation_rounds,
#             allocation_rounds,
#             intervention_rounds,
#             intervention_budget,
#             preferences_model,
#             allocation_model, 
#             intervention_model, 
#             preference_model_params=preference_model_params,
#             update_preference_params=update_preference_params)
    
#     dissimilarity_index.append(di[-1].mean())

# fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot(optimal_group_fractions, dissimilarity_index, marker='o', label=f'optimal_group_fraction={optimal_group_fraction}')
# ax.set_xlabel('optimal_group_fraction')
# ax.set_ylabel('dissimilarity_index')
# ax.set_ylim([0, 1])
# fig.show()

# #%% 3D

# preference_model_params = {
#     'c_weight': 0.2,
#     'M': 0.6,
#     'init_facility_composition': 'node',
# }
# update_preference_params = True

# runner = Runner(network, population, facilities, logger=None)

# c_weights = np.arange(0.1, 1, 0.1)
# optimal_group_fractions = np.arange(0.1, 1, 0.1)

# dissimilarity_index = []

# for c_weight in c_weights:
#     for optimal_group_fraction in optimal_group_fractions:
#         print(f'c_weight={c_weight}, optimal_group_fraction={optimal_group_fraction}')
#         preference_model_params['pop_optimal_grp_frac'] = float(optimal_group_fraction)
#         preference_model_params['c_weight'] = float(c_weight)

#         di = runner.run_simulation(
#                 simulation_rounds,
#                 allocation_rounds,
#                 intervention_rounds,
#                 intervention_budget,
#                 preferences_model,
#                 allocation_model, 
#                 intervention_model, 
#                 preference_model_params=preference_model_params,
#                 update_preference_params=update_preference_params)
        
#         dissimilarity_index.append([c_weight, optimal_group_fraction, di[-1].mean()])

# dissimilarity_index = np.array(dissimilarity_index)
# #%%
# fig = plt.figure(figsize = (5, 5))
# ax = plt.axes(projection ="3d")

# # ax.scatter(c_weights, optimal_group_fractions, dissimilarity_index, color = "green", marker = "o")
# ax.scatter(dissimilarity_index[:, 0], dissimilarity_index[:, 1], dissimilarity_index[:, 2], color = "green", marker = "o")

# # Set axis labels
# ax.set_xlabel('C_weight')
# ax.set_ylabel('optimal_group_fraction')
# ax.set_zlabel('dissimilarity_index')
# ax.set_zlim([0, 1])
# fig.show()

# #%% Line of the same chart
# fig, ax = plt.subplots(figsize = (5, 5))

# for c_w in c_weights:
#     values=  dissimilarity_index[dissimilarity_index[:, 0] == c_w]
#     ax.plot(values[:, 1], values[:, 2], marker='o', label=f'CW: {c_w}')

# ax.set_xlabel('optimal_group_fraction')
# ax.set_ylabel('dissimilarity_index')
# fig.legend()
# fig.show()
# # %%


# #%%
# ######### GRID #########
env = './envs/grid/GRID_10x10_[0.8]'

preferences_model = 'distance_composition'
allocation_model = 'random_serial_dictatorship'
intervention_model = 'none'
simulation_rounds = 30
intervention_rounds = 10
intervention_budget = 1
allocation_rounds = 5

preference_model_params = {
    'M': 0.6,
    'init_facility_composition': 'node',
}
update_preference_params = True

c_weights = np.arange(0.1, 1, 0.1)
optimal_group_fractions = np.arange(0.1, 1, 0.1)

dissimilarity_index = []

for c_weight in c_weights:
    for optimal_group_fraction in optimal_group_fractions:
        print(f'c_weight={c_weight}, optimal_group_fraction={optimal_group_fraction}')

        network = Network(f'{env}/network.gml', calc_tt_mx=True)
        population = pd.read_csv(f'{env}/population.csv')
        facilities = pd.read_csv(f'{env}/facilities.csv')
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
for i, c_w in enumerate(c_weights):
    values = dissimilarity_index[dissimilarity_index[:, 0] == c_w]
    ax.plot(values[:, 1], values[:, 2], marker=markers[i], label=f'CW: {round(c_w, 1)}')

ax.set_xlabel('optimal_group_fraction')
ax.set_ylabel('dissimilarity_index')
ax.set_ylim([0, 1.1])
fig.suptitle(env)
fig.legend(loc='lower right')
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

            network = Network(f'{env}/network.gml', calc_tt_mx=True)
            population = pd.read_csv(f'{env}/population.csv')
            facilities = pd.read_csv(f'{env}/facilities.csv')
            runner = Runner(network, population, facilities, logger=None)

            preference_model_params = {
                'M': 0.6,
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

def plot_di_progress_by_param(di, param1, param2, env, pref_model, alloc_model, inter_model, sim_rounds, inter_rounds, inter_budget, alloc_rounds, M, rounds_with_intervention, param1_name='cw', param2_name='ogf', colors=None, line_styles=None):
    fig, axs = plt.subplots(math.ceil(len(param1) / 3), 3, figsize = (10, 10))
    
    for i, p1 in enumerate(param1):
        for j, p2 in enumerate(param2):
            ax_i = i // 3
            ax_j = i % 3        
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
            axs[ax_i][ax_j].set_ylabel('dissimilarity_index')

            rwi = rounds_with_intervention[(rounds_with_intervention[:, 0] == p1) & (rounds_with_intervention[:, 1] == p2)]
            for r in rwi[0][2]:
                axs[ax_i][ax_j].axvline(r, linestyle='dashed', linewidth=1, color='#C2C2C2', alpha=0.1)

    handles, labels = axs[0][0].get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(f'{env} \n {pref_model} | {alloc_model} | {inter_model} \n {sim_rounds} simulation rounds, {inter_rounds} intervention rounds, {inter_budget} intervention budget, {alloc_rounds} allocation rounds, M={M}')
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
env = './envs/grid/GRID_10x10_[0.8]'

di, rwi = gen_and_plot(env=env,
            pref_model='distance_composition',
            alloc_model='random_serial_dictatorship',
            inter_model='closeness',
            sim_rounds=30,
            inter_rounds=5,
            inter_budget=1,
            alloc_rounds=5,
            M=0.6,
            c_weights=np.round(np.arange(0.1, 1, 0.1), 1),
            opt_group_frac=np.round(np.arange(0.1, 1, 0.1), 1))


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

            network = Network(f'{env}/network.gml', calc_tt_mx=True)
            population = pd.read_csv(f'{env}/population.csv')
            facilities = pd.read_csv(f'{env}/facilities.csv')
            runner = Runner(network, population, facilities, logger=None)

            preference_model_params = {
                'M': 0.6,
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

inter_models = ['none', 'random', 'closeness', 'group_closeness', 'betweenness', 'group_betweenness', 'degree', 'group_degree']

colors = {
    'none': 'gray', 'random': 'gray', 'closeness': '#e60049', 'group_closeness': '#e60049', 'betweenness': '#0bb4ff', 'group_betweenness': '#0bb4ff', 'degree': '#50e991', 'group_degree': '#50e991'
}
line_styles = {
    'none': '-.', 'random': '--', 'closeness': '-', 'group_closeness': '--', 'betweenness': '-', 'group_betweenness': '--', 'degree': '-', 'group_degree': '--'
}

c_weights = np.round(np.arange(0, 1, 0.1), 1)

di, rounds_with_intervention = di_progress_by_inter_model(
            env=env,
            pref_model='distance_composition',
            alloc_model='random_serial_dictatorship',
            inter_models=inter_models,
            sim_rounds=30,
            inter_rounds=10,
            inter_budget=5,
            alloc_rounds=5,
            c_weights=c_weights)

plot_di_progress_by_param(di, 
                        c_weights, 
                        inter_models, 
                        env=env,
                        pref_model='distance_composition', 
                        alloc_model='random_serial_dictatorship', 
                        inter_model=None,
                        sim_rounds=30,
                        inter_rounds=10,
                        inter_budget=5,
                        alloc_rounds=5,
                        M=0.6,
                        rounds_with_intervention=rounds_with_intervention,
                        param1_name='composition weight',
                        param2_name='inter_model',
                        colors=colors, 
                        line_styles=line_styles)
# %%
