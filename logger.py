import datetime
from pathlib import Path
import yaml

class Logger(object):
    def __init__(self, config):
        now = datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')
        self.results_path = Path('./results') / f"{now}_{config['preferences_model']}_{config['allocation_model']}_{config['intervention_model']}"
        self.results_path.mkdir(parents=True, exist_ok=True)
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

    def save_plot(self, fig, filename, round=None):
        """Saves the given figure to the results folder.

        Args:
            fig (matplotlib.figure.Figure): the figure to save.
            filename (str): the name of the file to save.
            round(str): if given, the file will be saved in the child rounds folder.
        """
        if round is None:
            fig.savefig(self.results_path / filename)
        else:
            path = self.results_path / 'rounds' / str(round)
            path.mkdir(parents=True, exist_ok=True)
            
            fig.savefig(path / filename)