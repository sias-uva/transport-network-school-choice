#%% This file is just for plotting results of multiple runs in one plot and creating other output-related plots.
# It is therefore meant to be used as a script and not as a module.
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

plt.rcParams.update({'font.size': 22})
TITLE_FONT_SIZE = 28
SUBTITLE_FONT_SIZE = 26
LEGEND_FONT_SIZE = 16

colors = {
    'none': 'gray', 
    'random': 'gray',
    'closeness': '#e60049', 
    'group_closeness': '#e60049', 
    'betweenness': '#0bb4ff', 
    'group_betweenness': '#0bb4ff', 
    'degree': '#50e991', 
    'group_degree': '#50e991',
    'move_facilities': '#0bb4ff'
}
line_styles = {
    'none': '-.', 
    'random': '--', 
    'closeness': '--',
    'group_closeness': '--', 
    'betweenness': '--',
    'group_betweenness': '--', 
    'degree': '-', 
    'group_degree': '--',
    'move_facilities': '-'
}

#%% Plot the simulation results side by sidex
def plot_results(results, ax, labels=None):
    # Names of the results
    result_path = Path('./results')

    # Confidence intervals of dissimilarity index
    diss_ci = []
    output_files = []
    for model_name in results.keys():
        diss_ci.append(np.loadtxt(result_path / f'{results[model_name]}/dissimilarity_index.txt', delimiter=','))

        with open(result_path / f'{results[model_name]}/output.txt') as f:
            d = yaml.safe_load(f)
        output_files.append(d)

    for i, model_name in enumerate(results.keys()):
        sim_rounds = diss_ci[i].shape[0]

        if labels:
            label_name = labels[model_name]
        else:
            labels = model_name

        ax.plot(range(sim_rounds), diss_ci[i][:, 0], line_styles[model_name], label=f'{label_name}', color=colors[model_name], linewidth=3)
        ax.fill_between(range(sim_rounds), diss_ci[i][:, 1], diss_ci[i][:, 2], color=colors[model_name], alpha=.1)
        
        for r in output_files[i]['rounds_with_intervention']:
                ax.axvline(r, linestyle='dashed', linewidth=1, color='#C2C2C2', alpha=0.1)

    return ax
    # fig.legend(loc='lower left', fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0, -.05), ncol=2)

# # results to plot
results_ams = {

        'none': '20240221_07_29_40.295559_grid_50_5_3_1_distance_composition_random_serial_dictatorship_none',
        'random': '20240221_11_11_31.632908_grid_50_5_3_1_distance_composition_random_serial_dictatorship_random_move_facilities',
        'closeness': '20240221_07_30_09.047804_grid_50_5_3_1_distance_composition_random_serial_dictatorship_closeness',
        'move_facilities': '20240221_10_52_50.153571_grid_50_5_3_1_distance_composition_random_serial_dictatorship_move_facilities'
}

labels = {
    'none':'none',
    'random': 'random',
    'move_facilities': 'targeted relocation',
    'closeness': 'closeness centrality'
}

# results to plot
# results_sbm = {
#         'none': '20240205_14_28_36.847198_amsterdam_neighborhoods_30_5_10_5_distance_composition_random_serial_dictatorship_none',
#         'closeness': '20240205_14_01_51.731862_amsterdam_neighborhoods_30_5_10_5_distance_composition_random_serial_dictatorship_closeness',
#         'betweenness': '20240205_13_47_43.562741_amsterdam_neighborhoods_30_5_10_5_distance_composition_random_serial_dictatorship_betweenness'        }


# results_grid = {
#         'none': '20230825_13_42_50.822429_grid_30_5_10_5_distance_composition_random_serial_dictatorship_none',
#         'random': '20230825_13_43_04.463970_grid_30_5_10_5_distance_composition_random_serial_dictatorship_random',
#         'closeness': '20230825_13_43_15.898486_grid_30_5_10_5_distance_composition_random_serial_dictatorship_closeness',
#         'group_closeness': '20230825_13_43_30.317528_grid_30_5_10_5_distance_composition_random_serial_dictatorship_group_closeness',
#         'betweenness': '20230825_13_44_51.884134_grid_30_5_10_5_distance_composition_random_serial_dictatorship_betweenness',
#         'group_betweenness': '20230825_13_45_00.741695_grid_30_5_10_5_distance_composition_random_serial_dictatorship_group_betweenness',
#         'degree': '20230825_13_45_33.405371_grid_30_5_10_5_distance_composition_random_serial_dictatorship_degree',
#         'group_degree': '20230825_13_45_41.156859_grid_30_5_10_5_distance_composition_random_serial_dictatorship_group_degree',
#
# }

fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
# get_figure(f"Average Dissimilarity Index in Schools",
#                         subtitle="30 simulation rounds, 15 intervention rounds",
#                         xlabel='Simulation round',
#                         ylabel='Dissimilarity Index',
#                         figsize=(12, 7.5),
#                         ylim=(0, 0.8))

# plot_results(results_sbm, ax[0])
plot_results(results_ams, ax, labels)
# plot_results(results_grid, ax[2])
# ax[0].set_ylim(0, 1.0)
ax.set_ylim(0, 1.0)
# ax[2].set_ylim(0, 0.9)
# ax[0].set_xlabel('Simulation round')
# ax[0].set_ylabel('Dissimilarity Index')
ax.set_xlabel('Simulation round')
ax.set_ylabel('Dissimilarity Index')
# ax[0].set_title(f'(A) Synthetic Environment', fontsize=SUBTITLE_FONT_SIZE)
# ax.set_title(f'Grid Environment', fontsize=SUBTITLE_FONT_SIZE)
# ax[2].set_title(f'(C) Grid Environment', fontsize=SUBTITLE_FONT_SIZE)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0),
          ncol=2, fancybox=True, shadow=True)



# fig.suptitle('Impact of Network Interventions on Segregation (TODO ADD GROUP BETWEENNESS)', fontsize=TITLE_FONT_SIZE)
# handles, labels = ax.get_legend_handles_labels()
# plt.figlegend(handles, labels, loc = 'lower center', ncol=4, labelspacing=0., bbox_to_anchor=(0.5, -0.15))
plt.savefig('relocation.png')
fig.show()


# %%
