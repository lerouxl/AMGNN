import copy
import logging
import wandb
import numpy as np

from dataloader import simulation_files
from utils.load_arc import load_arcs_list
from utils.save_arc import save_arc, load_arc


def preprocess_folder(simu: str, all_arc_files: list, tmp_arc_folder: str) -> None:
    """Launch the data preprocessing on an entire simulation.

    This function will list all simulations available by scanning the arc path for there second parents as the
    simulation folder. Then launch the pre precessing of the arc files.

    Parameters
    ----------
    simu: Path
        Path of the simulation folders.
    all_arc_files: list
        The list of all arc files. Its expected to have multiple simulation folders and where the name of the simulation
        folder is the second parents of the arc path.
    tmp_arc_folder: str
        Where to save the preprocessed files.

    Returns
    -------
    None
    """
    # for one simulation
    # extract all files for this simulation folder
    simu_arc_files = [p for p in all_arc_files if str(simu) == p.parents[2].stem]

    # Organise the files per simulation step
    step_files = simulation_files.organise_files(simu_arc_files)

    preprocess_files(step_files, tmp_arc_folder)


def preprocess_files(step_files: list, dest_folder: str) -> None:
    """ Do the pre processing of the arc files.

    Load the ARC csv files contained in the step_files list and process them to only keep the relevant data and store
    them in a specified format (float32 by default).
    The kept data are the voxel coordinate, and the features : "XDIS", "YDIS", "ZDIS" and "TEMPTURE".
    Displacement features such as XDIS,YDIS and ZDIS are converted in milimeter and temperature (TEMPTURE) is converted
    in celcius degrees.

    Then a pkl file is created in dest_folder, containing the kept data, the name of the previous step file, the previous
    step file and the original voxels coordinates (the voxel coordinate - the deformation).

    Parameters
    ----------
    step_files: list
        List of all path to the ARC csv files.
    dest_folder: str
        Where the temporary folder will be created.

    Returns
    -------
    None, preprocessed files are saved in the *dest_folder*.
    """
    for i, raw_path in enumerate(step_files):

        arc = load_arcs_list(raw_path, )
        arc.load_meta_parameters(
            increment_id=i, build_path=None, increments_path=None
        )
        # Convert data from meter to mm
        arc.coordinate = arc.coordinate * 1000
        arc.data.XDIS, arc.data.YDIS, arc.data.ZDIS = arc.data.XDIS*1000, arc.data.YDIS*1000, arc.data.ZDIS*1000

        # Convert from Kelvin to Celsius
        arc.data.TEMPTURE = arc.data.TEMPTURE - 273.15

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

        # Add the process step to the data:
        arc.data.subProcessName = np.full_like(arc.data.XDIS,
                                               f"{arc.metaparameters.subProcessName}", dtype="S120")
        try:
            process_name = str(arc.metaparameters.subProcessName, "utf-8")
        except:
            process_name = str(arc.metaparameters.subProcessName)

        process_category, process_features = subprocessname_to_cat_features(process_name)

        # Pass those features to all nodes
        arc.data.process_category = np.broadcast_to(np.array([process_category], dtype=float),
                                                    [arc.data.XDIS.shape[0], 6])
        arc.data.process_features = np.full_like(arc.data.XDIS, process_features, dtype=float)

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


def subprocessname_to_cat_features(process_name: str) -> tuple[list[int], int]:
    """Transform the subProcessName in a category and a features.

    The subProcessName will be classified into 6 categories.
    Categories:
        - AM_Layer : [1,0,0,0,0,0]
        - process : [0,1,0,0,0,0]
        - Postcooling: [0,0,1,0,0,0]
        - Powderremoval : [0,0,0,1,0,0]
        - Unclamping : [0,0,0,0,1,0]
        - Cooling-1 : [0,0,0,0,0,1]

    If the subProcessName is an AM_layer, then the layer number will be extracted and used as features.
    Examples:
        - "AM_Layer 1" -> feature = 1
        - "AM_Layer 10" -> feature = 10
        - "AM_Layer 31" -> feature = 31

    Note: The byte object of subProcessName should be converted into a string, with for example
        str(subProcessName, "utf-8")

    Parameters
    ----------
    process_name: str
        String of the subProcessName. Not the byte.

    Returns
    -------
    cat: list[int]
        list of int used as an one hot encoder for the type of the subProcess.
    features: int
        0 if the subProcess is not an 'AM_Layer', if it is, the number of the 'AM_Layer'.
        Warning, this values is not normalised.
    """
    # Initialise the category and features variables
    cat = [0, 0, 0, 0, 0, 0]  # AM_Layer, process, Postcooling, Powderremoval, Unclamping, Cooling-1
    feature = 0

    log = logging.getLogger(__name__)

    # if we are in a layer step, we extract the layer number as feature
    if "AM_Layer" in process_name:
        feature = int((process_name.split(" ")[-1]))
        cat = [1, 0, 0, 0, 0, 0]
    # Else, the feature stay at 0 and we change the process category
    elif "process" in process_name:
        cat = [0, 1, 0, 0, 0, 0]
    elif "Postcooling" in process_name:
        cat = [0, 0, 1, 0, 0, 0]
    elif "Powderremoval" in process_name:
        cat = [0, 0, 0, 1, 0, 0]
    elif "Unclamping" in process_name:
        cat = [0, 0, 0, 0, 1, 0]
    elif "Cooling-1" in process_name:
        cat = [0, 0, 0, 0, 0, 1]
    # In case of unknow categorie, raisen an error
    if cat == [0, 0, 0, 0, 0, 0]:
        log.debug( f"Unknow categorie for {process_name}")
        #raise f"Unknow categorie for {process_name}"

    return cat, feature
