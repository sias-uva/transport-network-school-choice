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
