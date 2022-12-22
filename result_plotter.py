#%% This file is just for plotting results of multiple runs in one plot and creating other output-related plots.
# It is therefore meant to be used as a script and not as a module.
import numpy as np
from pathlib import Path
from plot import get_figure

# results to plot
results = ['20221222_16_38_01.903090_sbm_toy_model_random_serial_dictatorship_random',
           '20221222_16_27_44.819029_sbm_toy_model_random_serial_dictatorship_closeness',       
           '20221222_16_49_59.724805_sbm_toy_model_random_serial_dictatorship_betweenness',
           '20221222_16_51_46.671943_sbm_toy_model_random_serial_dictatorship_degree']

# Names of the results
names = ['random', 'closeness', 'betweenness', 'degree']
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
                    ylim=(0, 1))

for i in range(len(results)):
    sim_rounds = diss_ci[i].shape[0]

    ax.plot(range(sim_rounds), diss_ci[i][:, 0], label=f'{names[i]}')
    ax.fill_between(range(sim_rounds), diss_ci[i][:, 1], diss_ci[i][:, 2], color=f'C{i}', alpha=.1)

fig.legend()
# %%
