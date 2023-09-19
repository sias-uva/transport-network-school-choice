import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import yaml
import igraph as ig

class Logger(object):
    def __init__(self, config):
        now = datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')
        environment = config['network_file'].split('/')[2]
        self.results_path = Path('./results') / f"{now}_{environment}_{config['simulation_rounds']}_{config['allocation_rounds']}_{config['intervention_rounds']}_{config['intervention_budget']}_{config['preferences_model']}_{config['allocation_model']}_{config['intervention_model']}"
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.rounds_path = self.results_path / 'rounds'
        self.rounds_path.mkdir(parents=True, exist_ok=True)
        self.output_file = self.results_path / 'output.txt'

        with open(self.output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def append_to_output_file(self, text: str):
        """Appends given text to the output file.

        Args:
            text (str): the text to append.
        """
        with open(self.output_file, 'a') as f:
            f.write(text)
            f.write('\n')

    def log_dataframe(self, dataframe: pd.DataFrame, filename: str, round=None):
        """Saves a given pandas dataframe to the results folder.

        Args:
            text (str): the text to append.
        """
        if round is None:
            dataframe.to_csv(path, index=False)
        else: 
            path = self.rounds_path / str(round)
            path.mkdir(parents=True, exist_ok=True)

            dataframe.to_csv(path / filename, index=False)

    def log_numpy_array(self, array: np.array, filename: str, round=None):
        """Saves a given numpy array to the results folder.

        Args:
            text (str): the text to append.
        """
        path = self.results_path / filename
        if round is not None:
            path = self.rounds_path / str(round) / filename

        np.savetxt(path, array, delimiter=',')

    def save_plot(self, fig, filename, round=None, subdir=None):
        """Saves the given figure to the results folder.

        Args:
            fig (matplotlib.figure.Figure): the figure to save.
            filename (str): the name of the file to save.
            round(str): if given, the file will be saved in the child rounds folder. 
        """
        path = self.results_path
        if round is not None:
            path = self.rounds_path / str(round)
        if subdir is not None:
            path = path / subdir
        
        path.mkdir(parents=True, exist_ok=True)

        fig.savefig(path / filename)

        # if round is None:
        #     fig.savefig(self.results_path / filename)
        # else:
        #     path = self.rounds_path / str(round)
        #     path.mkdir(parents=True, exist_ok=True)
            
        #     fig.savefig(path / filename)
        plt.close(fig)

    def save_igraph_plot(self, network, filename='network.pdf', edges_to_color=None, facilities_to_label=None, round=None, vertex_size=10, vertex_label_size=7):
        """Saves a given igraph plot to the results folder.

        Args:
            network (network): The network to save an image of.
            filename (str, optional): the name of the file to save. Defaults to 'network.pdf'.
            edges_to_color (edge or list of edges, optional): an edge or a list of edges (ig.Edge to color). Defaults to None.
            round (int, optional): round of the simulation - gets appended to the filename and title. Defaults to None.
            vertex_size (int, optional): size of the vertices to plot. Defaults to 10.
            vertex_label_size (int, optional): size of the vertex labels to plot. Defaults to 5.
        """
        # TODO: Instead of doing this maybe create a new copy of the network and plot that? Consider memory constraints.
        if edges_to_color is not None:
            if type(edges_to_color) is not list: edges_to_color = [ edges_to_color ]
            for i, e in enumerate(edges_to_color):
                e['color'] = 'blue'
                e['label'] = i + 1

        if facilities_to_label is not None:
            for v in network.network.vs:
                if np.isin(v['id'], facilities_to_label):
                    v['label'] = str(v['label']) + '*'
        
        if round is None:
            path = self.results_path / filename
        else:
            path = self.rounds_path / str(round)
            path.mkdir(parents=True, exist_ok=True)
            path = path / filename
        
        network.network.vs['label_size'] = vertex_label_size
        if 'x' in network.network.vs.attribute_names():
            layout=[(v['y'], v['x']) for v in network.network.vs]
        elif 'GRID' in network.network_path and 'lattice' not in network.network_path:
            layout=network.network.layout("grid")
        else:
            layout=None

        ig.plot(network.network,
                target=path, 
                layout=layout if layout is not None else network.network_layout, 
                vertex_size=vertex_size)

        if edges_to_color is not None:
            for e in edges_to_color:
                e['color'] = network.DEFAULT_EDGE_COLOR
                e['label'] = None
