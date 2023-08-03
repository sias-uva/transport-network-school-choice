#%%
from runner import Runner
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from network import Network

######### SBM #########
#%% Here we test what happens in a non-segregated environment, when we adapt the optimal group fraction parameter (homophily).

# env = './envs/sbm/SBM_6_6_0.7_0.01_pop_1000_maj_pop_pct_0.5'
env = './envs/sbm/SBM_6_6_0.7_0.01_pop_1000_maj_pop_pct_0.6_0.8_0.9'

network = Network(f'{env}/network.gml', calc_tt_mx=True)
population = pd.read_csv(f'{env}/population.csv')
facilities = pd.read_csv(f'{env}/facilities.csv')
preferences_model = 'distance_composition'
allocation_model = 'random_serial_dictatorship'
intervention_model = 'none'
simulation_rounds = 30
intervention_rounds = 2
intervention_budget = 1
allocation_rounds = 5

preference_model_params = {
    'c_weight': 0.1,
    'M': 0.6,
    'init_facility_composition': 'node',
    'pop_optimal_grp_frac': None
}
update_preference_params = True


runner = Runner(network, population, facilities, logger=None)

optimal_group_fractions = np.arange(0.1, 1, 0.1)

dissimilarity_index = []

for optimal_group_fraction in optimal_group_fractions:
    preference_model_params['pop_optimal_grp_frac'] = float(optimal_group_fraction)

    di = runner.run_simulation(
            simulation_rounds,
            allocation_rounds,
            intervention_rounds,
            intervention_budget,
            preferences_model,
            allocation_model, 
            intervention_model, 
            preference_model_params=preference_model_params,
            update_preference_params=update_preference_params)
    
    dissimilarity_index.append(di[-1].mean())

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(optimal_group_fractions, dissimilarity_index, marker='o', label=f'optimal_group_fraction={optimal_group_fraction}')
ax.set_xlabel('optimal_group_fraction')
ax.set_ylabel('dissimilarity_index')
ax.set_ylim([0, 1])
fig.show()

#%%
env = './envs/sbm/SBM_6_6_0.7_0.01_pop_1000_maj_pop_pct_0.5'
env = './envs/sbm/SBM_6_6_0.7_0.01_pop_1000_maj_pop_pct_0.6_0.8_0.9'

network = Network(f'{env}/network.gml', calc_tt_mx=True)
population = pd.read_csv(f'{env}/population.csv')
facilities = pd.read_csv(f'{env}/facilities.csv')
preferences_model = 'distance_composition'
allocation_model = 'random_serial_dictatorship'
intervention_model = 'none'
simulation_rounds = 30
intervention_rounds = 2
intervention_budget = 1
allocation_rounds = 5

preference_model_params = {
    'M': 0.6,
    'init_facility_composition': 'node',
    'pop_optimal_grp_frac': 0.5
}
update_preference_params = True


runner = Runner(network, population, facilities, logger=None)

c_weights = np.arange(0.1, 1, 0.1)

dissimilarity_index = []

for c_weight in c_weights:
    preference_model_params['c_weight'] = float(c_weight)

    di = runner.run_simulation(
            simulation_rounds,
            allocation_rounds,
            intervention_rounds,
            intervention_budget,
            preferences_model,
            allocation_model, 
            intervention_model, 
            preference_model_params=preference_model_params,
            update_preference_params=update_preference_params)
    
    dissimilarity_index.append(di[-1].mean())

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(c_weights, dissimilarity_index, marker='o')
ax.set_xlabel('c_weight')
ax.set_ylabel('dissimilarity_index')
ax.set_ylim([0, 1])
fig.show()

#%%
env = './envs/sbm/SBM_6_6_0.7_0.01_pop_1000_maj_pop_pct_0.6_0.8_0.9'

network = Network(f'{env}/network.gml', calc_tt_mx=True)
population = pd.read_csv(f'{env}/population.csv')
facilities = pd.read_csv(f'{env}/facilities.csv')
preferences_model = 'distance_composition'
allocation_model = 'random_serial_dictatorship'
intervention_model = 'none'
simulation_rounds = 30
intervention_rounds = 2
intervention_budget = 1
allocation_rounds = 5

preference_model_params = {
    'M': 0.6,
    'init_facility_composition': 'node',
}
update_preference_params = True


runner = Runner(network, population, facilities, logger=None)

c_weights = np.arange(0.1, 1, 0.1)
optimal_group_fractions = np.arange(0.1, 1, 0.1)

dissimilarity_index = []

for c_weight in c_weights:
    for optimal_group_fraction in optimal_group_fractions:
        preference_model_params['pop_optimal_grp_frac'] = float(optimal_group_fraction)
        preference_model_params['c_weight'] = float(c_weight)

        di = runner.run_simulation(
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

#%%
# Creating figure
dissimilarity_index = np.array(dissimilarity_index)
fig = plt.figure(figsize = (5, 5))
ax = plt.axes(projection ="3d")

# ax.scatter(c_weights, optimal_group_fractions, dissimilarity_index, color = "green", marker = "o")
ax.scatter(dissimilarity_index[:, 0], dissimilarity_index[:, 1], dissimilarity_index[:, 2], color = "green", marker = "o")

# Set axis labels
ax.set_xlabel('C_weight')
ax.set_ylabel('optimal_group_fraction')
ax.set_zlabel('dissimilarity_index')
ax.set_zlim([0, 1])
fig.show()

# %%
