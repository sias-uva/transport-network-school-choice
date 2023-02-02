#%% This file is just for plotting results of multiple runs in one plot and creating other output-related plots.
# It is therefore meant to be used as a script and not as a module.
import numpy as np
from pathlib import Path
from plot import get_figure

# results to plot
# results = ['20221223_17_04_58.447799_sbm_toy_model_random_serial_dictatorship_random',
#            '20221223_17_05_11.352955_sbm_toy_model_random_serial_dictatorship_closeness',       
#            '20221223_17_05_22.281079_sbm_toy_model_random_serial_dictatorship_betweenness',
#            '20221223_17_05_42.548594_sbm_toy_model_random_serial_dictatorship_degree',
#            '20230117_17_24_48.556593_sbm_toy_model_random_serial_dictatorship_group_closeness',
#            '20230117_17_25_01.117971_sbm_toy_model_random_serial_dictatorship_group_betweenness',
#            '20230117_17_25_29.002672_sbm_toy_model_random_serial_dictatorship_group_degree']

results = ['20230130_12_27_43.667620_amsterdam_neighborhoods_distance_popularity_random_serial_dictatorship_none',
           '20230130_12_34_52.485755_amsterdam_neighborhoods_distance_popularity_random_serial_dictatorship_random',       
           '20230130_12_46_01.100541_amsterdam_neighborhoods_distance_popularity_random_serial_dictatorship_closeness',
           '20230130_12_46_31.355849_amsterdam_neighborhoods_distance_popularity_random_serial_dictatorship_betweenness',
           '20230130_12_46_55.147997_amsterdam_neighborhoods_distance_popularity_random_serial_dictatorship_degree',
           '20230130_12_47_16.141056_amsterdam_neighborhoods_distance_popularity_random_serial_dictatorship_group_closeness',
           '20230130_13_27_05.602647_amsterdam_neighborhoods_distance_popularity_random_serial_dictatorship_group_betweenness',
           '20230130_12_54_42.999950_amsterdam_neighborhoods_distance_popularity_random_serial_dictatorship_group_degree']

# Names of the results
names = ['none', 'random', 'closeness', 'betweenness', 'degree', 'group_closeness', 'group_betweenness', 'group_degree']
line_styles = ['--', '--', '--', '--', '--', '-', '-', '-']
result_path = Path('./results')

# Confidence intervals of dissimilarity index
diss_ci = []
output_files = []
for result in results:
    diss_ci.append(np.loadtxt(result_path / f'{result}/dissimilarity_index.txt', delimiter=','))

    d = {}
    with open(result_path / f'{result}/output.txt') as f:
        for line in f:
            (key, val) = line.split(':')
            d[key] = val.strip()
    output_files.append(d)

#%%
fig, ax = get_figure(f"Dissimilarity Index",
                    xlabel='Simulation round',
                    ylabel='Dissimilarity Index',
                    figsize=(7, 5),
                    ylim=(0, 0.5))

for i in range(len(results)):
    sim_rounds = diss_ci[i].shape[0]

    ax.plot(range(sim_rounds), diss_ci[i][:, 0], line_styles[i], label=f'{names[i]}')
    ax.fill_between(range(sim_rounds), diss_ci[i][:, 1], diss_ci[i][:, 2], color=f'C{i}', alpha=.1)

fig.legend()
# %%
