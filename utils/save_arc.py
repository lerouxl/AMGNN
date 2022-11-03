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
    coordinate = arc.original_coordinate
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

    # first step
    is_first_step = np.asarray(arc.is_first_step)

    if arc.previous_file_name is None:
        previous_file_name = np.asarray("")
    else:
        previous_file_name = np.asarray(arc.previous_file_name)



    # If the arc object was displayed, it cannot be pickled
    with open(file_path, "wb") as file:
        np.savez(file,
                 is_first_step = is_first_step,
                 connectivity=connectivity,
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
                 points_types =points_types,
                 previous_file_name=previous_file_name,

                 )
    if file_path.is_file():
        logging.info(f"Written {file_path} successfully.")
    else:
        logging.error(f"{file_path} was not correctly witted")
        raise FileNotFoundError(file_path.name)


def load_arc(file: Union[Path, str], load_past_arc= True):
    """Load npz arc file.

    Load an npz arc save file and create the corresponding arc object
    Parameters
    ----------
    file: Union[Path, str]
        Path to the npz file (with extension)
    load_past_arc: bool
        Load the arc specified in arc.previous_file_name.
        We are supposing both file are in the same folder.
    Returns
    -------
    Arc_reader: The recreated arc object.
    """
    file = Path(file)
    # Check file is existing
    if not file.is_file():
        msg = f"Error opening the file {file}. This is not an existing file"
        logging.error(msg)
        raise FileNotFoundError( f"{file.name}")
    with np.load(file) as arrays:
        arc = Arc_reader(name="merged")

        # Extract first step value
        arc.is_first_step = arrays["is_first_step"].all()
        # Extract connectivity
        arc.connectivity = arrays["connectivity"]
        # Extract edge_index
        arc.edge_index = arrays["edge_index"].tolist()
        # Extract coordinate
        arc.coordinate = arrays["coordinate"]
        # Extract data
        arc.data.TEMPTURE = arrays["temperature"]
        arc.data.XDIS = arrays["xdis"]
        arc.data.YDIS = arrays["ydis"]
        arc.data.ZDIS = arrays["zdis"]
        arc.data.process_category = arrays["process_cat"]
        arc.data.process_features = arrays["process_features"]
        # Extract metaparameters
        arc.metaparameters.initialTemperature_C = float(arrays["initT"])
        arc.metaparameters.layerThickness_m = float(arrays["layerT"])
        arc.metaparameters.power_W = float(arrays["power"])
        arc.metaparameters.process_step = float(arrays["step"])
        arc.metaparameters.speed_m_s = float(arrays["speed"])
        arc.metaparameters.time_steps_length_s = float(arrays["time_step_lenght"])
        arc.metaparameters.time_steps_s = float(arrays["time_steps_s"])

        # Extract points types
        arc.points_types = arrays["points_types"]

        arc.previous_file_name = str(arrays["previous_file_name"])

    # Little hack to avoid numpy error when saving a None value
    if arc.previous_file_name == "":
        arc.previous_arc = None
        arc.previous_file_name = None

    if load_past_arc:
        if not arc.previous_file_name is None:
            past_file_path = (Path(file.parent) / arc.previous_file_name).with_suffix(".npz")
            arc.previous_arc = load_arc(past_file_path, False)
    else:
        pass


    logging.info(f"Loaded {file} successfully")
    return arc
