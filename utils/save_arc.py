import pickle
from simufact_arc_reader.ARC_CSV import Arc_reader
from pathlib import Path
from typing import Union
import logging
import numpy as np


def save_arc(arc: Arc_reader, root_folder: Union[Path, str], name: str):
    """Save the useful features of a the arc object,

    Save the variable of an Arc_reader file so they can be loader in a next step.
    The file are saved as numpy uncompressed array.
    Parameters
    ----------
    arc: Arc_reader
        The arc file to save.
    root_folder:  Union[Path, str]
        The path to the folder to save the file.
    name:
        Name of the file (without extension)

    Returns
    -------
    None
    """
    root_folder = Path(root_folder)
    # if folder do not exist, create it
    root_folder.mkdir(parents=True, exist_ok=True)
    file_path = Path(root_folder) / f"{name}.npz"

    # Extract connectivity
    connectivity = arc.connectivity.astype(int)
    # Extract edge_index
    edge_index = np.asarray(arc.edge_index)
    # Extract coordinate
    coordinate = arc.coordinate
    # Extract data
    temperature = arc.data.TEMPTURE
    xdis = arc.data.XDIS
    ydis = arc.data.YDIS
    zdis = arc.data.ZDIS
    process_cat = arc.data.process_category
    process_features = arc.data.process_features
    # Extract metaparameters
    initT = arc.metaparameters.initialTemperature_C
    layerT = arc.metaparameters.layerThickness_m
    power = arc.metaparameters.power_W
    step = arc.metaparameters.process_step
    speed = arc.metaparameters.speed_m_s
    time_step_lenght = arc.metaparameters.time_steps_length_s
    time_steps_s = arc.metaparameters.time_steps_s

    # Extract points types
    points_types = arc.points_types



    # If the arc object was displayed, it cannot be pickled
    with open(file_path, "wb") as file:
        np.savez(file, connectivity=connectivity,
                 edge_index=edge_index,
                 coordinate=coordinate,
                 temperature=temperature,
                 xdis=xdis,
                 ydis=ydis,
                 zdis=zdis,
                 process_cat=process_cat,
                 process_features=process_features,
                 initT=initT,
                 layerT=layerT,
                 power=power,
                 step=step,
                 speed=speed,
                 time_step_lenght=time_step_lenght,
                 time_steps_s=time_steps_s,
                 points_types =points_types
                 )
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
