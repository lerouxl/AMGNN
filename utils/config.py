import yaml
from pathlib import Path
import wandb

"""
Read the configuration file in configs/. 
The config file is used to parametrise the data preprocessing.
"""


def read_config(config_path: Path) -> dict:
    """Load the parameters form all the yml files (except sweep.yml) in config_path.
     The path to the yml file is taken from config_path.

     :param config_path: Path folder where are all the yml files

     Return a dictionary."""
    data_config = dict()

    for yml_file in config_path.glob("*.yml"):
        if yml_file.stem == "sweep":
            continue
        with open(yml_file, 'r') as file:
            file_dict = yaml.safe_load(file)
        data_config = data_config | file_dict  #: merge the 2 dictionaries

    return data_config


if __name__ == "__main__":
    configuration = read_config(Path("configs"))
    wandb.init(mode="offline", config=configuration)

    print(wandb.config)
