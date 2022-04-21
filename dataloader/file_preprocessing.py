import copy
import logging
import wandb
import numpy as np

from dataloader import simulation_files
from utils.load_arc import load_arcs_list
from utils.save_arc import save_arc, load_arc


def preprocess_folder(simu: str, all_arc_files: list, tmp_arc_folder: str) -> None:
    """
    Launch the data preprocessing on an entire simulation.
    :param simu:  Path of the simulation folder
    :param all_arc_files: list od all arc files
    :param tmp_arc_folder: where to save the preprocessed files.
    :return:
    """
    # for one simulation
    # extract all files for this simulation folder
    simu_arc_files = [p for p in all_arc_files if str(simu) == p.parents[3].stem]

    # Organise the files per simulation step
    step_files = simulation_files.organise_files(simu_arc_files)

    preprocess_files(step_files, tmp_arc_folder)


def preprocess_files(step_files: list, dest_folder: str) -> None:
    """
    Load the ARC csv files contained in the step_files list and process them to only keep the relevant data and store
    them in a specified format (float32 by default).
    The kept data are the voxel coordinate, and the features : "XDIS", "YDIS", "ZDIS" and "TEMPTURE".

    Then a pkl file is created in dest_folder, containing the kept data, the name of the previous step file, the previous
    step file and the original voxels coordinates (the voxel coordinate - the deformation).

    :param dest_folder: where the temporary folder will be created.
    :param step_files: List of all path to the ARC csv files.
    :return:
    """
    for i, raw_path in enumerate(step_files):

        arc = load_arcs_list(raw_path, )
        arc.load_meta_parameters(
            increment_id=i, build_path=None, increments_path=None
        )

        arc = load_arcs_list(raw_path)
        arc.load_meta_parameters(
            increment_id=i, build_path=None, increments_path=None
        )

        # Change data type to remove unecesary precision

        try:
            # Try to get the memory type from wandb
            float_type = wandb.config.float_type
        except:
            float_type = "float32"
        arc.coordinate = arc.coordinate.astype(float_type)
        arc.data.XDIS, arc.data.YDIS, arc.data.ZDIS = arc.data.XDIS.astype(float_type), arc.data.YDIS.astype(
            float_type), arc.data.ZDIS.astype(float_type)
        arc.data.TEMPTURE = arc.data.TEMPTURE.astype(float_type)

        # Clear raw_data
        arc.raw_data = None

        # Clear data
        for dat in dir(arc.data):
            if (dat in ["XDIS", "YDIS", "ZDIS", "TEMPTURE"]) or dat.startswith("__"):
                pass
            else:
                setattr(arc.data, dat, None)

        arc.original_coordinate = arc.coordinate - np.stack([arc.data.XDIS, arc.data.YDIS, arc.data.ZDIS],
                                                            axis=1)
        arc.original_coordinate = arc.original_coordinate.astype(float_type)

        # Previous arc file
        if i == 0:
            # If this is the initialisation file
            previous_file_name = None
            is_first_step = True
        else:
            # For the next simulation files
            folder_name = simulation_files.extract_the_simulation_folder(step_files[i - 1][0])
            step_ = simulation_files.extract_step_folder(step_files[i - 1][0])
            previous_file_name = f"{folder_name}_at_step_{step_}"
            is_first_step = False
            arc.previous_arc = previous_arc
        arc.previous_file_name = previous_file_name  # Gave the name of the previous file
        arc.is_first_step = is_first_step
        step_name = f"{simulation_files.extract_the_simulation_folder(raw_path[0])}_at_step_{simulation_files.extract_step_folder(raw_path[0])}"
        save_arc(arc, dest_folder, step_name)

        previous_arc = copy.deepcopy(arc)
