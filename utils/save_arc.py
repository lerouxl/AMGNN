import pickle
from simufact_arc_reader.ARC_CSV import Arc_reader
from pathlib import Path
from typing import Union
import logging


def save_arc(arc: Arc_reader, root_folder: Union[Path, str], name: str):
    """

    :param arc:
    :param root_folder:
    :param name:
    :return:
    """
    root_folder = Path(root_folder)
    # if folder do not exist, create it
    root_folder.mkdir(parents=True, exist_ok=True)
    file_path = Path(root_folder) / f"{name}.pkl"

    # If the arc object was displayed, it cannot be pickled
    with open(file_path, "wb") as file:
        pickle.dump(arc, file)
    if file_path.is_file():
        logging.info(f"Written {file_path} successfully.")
    else:
        logging.error(f"{file_path} was not correctly witted")
        raise FileNotFoundError(file_path.name)


def load_arc(file: Union[Path, str]):
    """

    :param file:
    :return:
    """
    file = Path(file)
    # Check file is existing
    if not file.is_file():
        msg = f"Error opening the file {file}. This is not an existing file"
        logging.error(msg)
        raise FileNotFoundError(file.name)
    with open(file, 'rb') as pickle_file:
        arc = pickle.load(pickle_file)
    logging.info(f"Loaded {file} successfully")
    return arc
