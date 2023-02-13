#%% This file is just for plotting results of multiple runs in one plot and creating other output-related plots.
# It is therefore meant to be used as a script and not as a module.
import numpy as np
from pathlib import Path
from plot import get_figure
import matplotlib.pyplot as plt
import yaml

plt.rcParams.update({'font.size': 22})
TITLE_FONT_SIZE = 28
SUBTITLE_FONT_SIZE = 26
LEGEND_FONT_SIZE = 16

def plot_results(results, ax):
    # Names of the results
    # names = ['none', 'random', 'closeness', 'betweenness', 'degree', 'group_closeness', 'group_betweenness', 'group_degree']
    names = ['none', 'random', 'closeness', 'group_closeness', 'betweenness', 'degree', 'group_degree']
    colors = ['gray', 'gray', '#e60049', '#e60049', '#0bb4ff', '#50e991', '#50e991']
    # line_styles = ['--', '--', '--', '--', '--', '-', '-', '-']
    line_styles = ['-.', '--', '--', '-', '--', '--', '-']
    result_path = Path('./results')

    # Confidence intervals of dissimilarity index
    diss_ci = []
    output_files = []
    for result in results:
        diss_ci.append(np.loadtxt(result_path / f'{result}/dissimilarity_index.txt', delimiter=','))

        with open(result_path / f'{result}/output.txt') as f:
            d = yaml.safe_load(f)
        output_files.append(d)

    # fig, ax_ = get_figure(f"Average Dissimilarity Index in Schools",
    #                     # subtitle="30 simulation rounds, 15 intervention rounds",
    #                     xlabel='Simulation round',
    #                     ylabel='Dissimilarity Index',
    #                     figsize=(12, 7.5),
    #                     ylim=(0, 0.8))


    for i in range(len(results)):
        sim_rounds = diss_ci[i].shape[0]

        ax.plot(range(sim_rounds), diss_ci[i][:, 0], line_styles[i], label=f'{names[i]}', color=colors[i], linewidth=3)
        ax.fill_between(range(sim_rounds), diss_ci[i][:, 1], diss_ci[i][:, 2], color=colors[i], alpha=.1)
        
        for r in output_files[i]['rounds_with_intervention']:
                ax.axvline(r, linestyle='dashed', linewidth=1, color='#C2C2C2', alpha=0.1)

    return ax
    # fig.legend(loc='lower left', fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0, -.05), ncol=2)

# # results to plot
results_ams = [
        '20230213_11_22_58.311840_amsterdam_neighborhoods_50_5_25_1_distance_composition_random_serial_dictatorship_none',
        '20230213_11_23_05.508838_amsterdam_neighborhoods_50_5_25_1_distance_composition_random_serial_dictatorship_random',
        '20230213_11_23_16.596338_amsterdam_neighborhoods_50_5_25_1_distance_composition_random_serial_dictatorship_closeness',
        '20230213_11_28_21.533442_amsterdam_neighborhoods_50_5_25_1_distance_composition_random_serial_dictatorship_group_closeness',
        '20230213_11_23_27.183322_amsterdam_neighborhoods_50_5_25_1_distance_composition_random_serial_dictatorship_betweenness',
        '20230213_11_24_02.853325_amsterdam_neighborhoods_50_5_25_1_distance_composition_random_serial_dictatorship_degree',
        '20230213_11_28_34.792581_amsterdam_neighborhoods_50_5_25_1_distance_composition_random_serial_dictatorship_group_degree'
        ]

# results to plot
results_sbm = [
        '20230210_17_20_32.969242_sbm_50_5_2_1_distance_composition_random_serial_dictatorship_none',
        '20230210_17_20_38.371774_sbm_50_5_2_1_distance_composition_random_serial_dictatorship_random',
        '20230210_17_19_45.202179_sbm_50_5_2_1_distance_composition_random_serial_dictatorship_closeness',
        '20230210_17_21_39.629291_sbm_50_5_2_1_distance_composition_random_serial_dictatorship_group_closeness',
        '20230210_17_21_20.236519_sbm_50_5_2_1_distance_composition_random_serial_dictatorship_betweenness',
        '20230210_17_21_27.596022_sbm_50_5_2_1_distance_composition_random_serial_dictatorship_degree',
        '20230210_17_22_25.239575_sbm_50_5_2_1_distance_composition_random_serial_dictatorship_group_degree',
        ]

fig, ax = plt.subplots(1, 2, figsize=(24, 7.5))
# get_figure(f"Average Dissimilarity Index in Schools",
#                         subtitle="30 simulation rounds, 15 intervention rounds",
#                         xlabel='Simulation round',
#                         ylabel='Dissimilarity Index',
#                         figsize=(12, 7.5),
#                         ylim=(0, 0.8))

plot_results(results_sbm, ax[0])
plot_results(results_ams, ax[1])
ax[0].set_ylim(0, 0.8)
ax[1].set_ylim(0, 0.8)
ax[0].set_xlabel('Simulation round')
ax[0].set_ylabel('Disimilarity Index')
ax[1].set_xlabel('Simulation round')
ax[1].set_ylabel('Disimilarity Index')
ax[0].set_title(f'(A) Synthetic Environment', fontsize=SUBTITLE_FONT_SIZE)
ax[1].set_title(f'(B) Amsterdam Environment', fontsize=SUBTITLE_FONT_SIZE)

# fig.suptitle('Impact of Network Interventions on Segregation (TODO ADD GROUP BETWEENNESS)', fontsize=TITLE_FONT_SIZE)
handles, labels = ax[0].get_legend_handles_labels()
plt.figlegend(handles, labels, loc = 'lower center', ncol=8, labelspacing=0., bbox_to_anchor=(0.5, -0.15))
fig.show()

# %%
