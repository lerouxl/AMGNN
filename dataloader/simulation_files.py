from typing import Union, List
from pathlib import Path


def organise_files(raw_files: List[Path]) -> List:
    """
    Take the list of raw files and join the file of the same simulation step together in a list of list.
    :param raw_files: List of all files, can be str of Path data
    :return:
    """
    raw_files = remove_baseplate_from_list(raw_files)
    simulation_folds = extract_simulation_folder(raw_files)
    # Create a dict to organise the files
    # |_ simulation_folder
    #                       |_ 000X
    #                               |_part.csv / supports.csv
    #                       |_ 000Y
    #                               |_part.csv

    #: Dict of Dict containing the list of files per simulation and step
    files_dictionary = dict()
    #: Dict of List of all found files per simulation folder, used for processing
    files_by_folder = dict()
    for fold in simulation_folds:
        files_dictionary[fold] = dict()
        files_by_folder[fold] = list()

    for file in raw_files:
        step = extract_step_folder(file)
        folder = extract_the_simulation_folder(file)

        if step in files_dictionary[folder]:
            files_dictionary[folder][step].append(file)
        else:
            files_dictionary[folder][step] = [file]

    organised_list = list()
    for key, value in files_dictionary.items():
        # Value has all the step of a folder key

        for key2, value2 in files_dictionary[key].items():
            # value 2 has all the path of a step
            organised_list.append(list)
            organised_list[-1] = value2

    return organised_list


def extract_step_folder(raw_file: Union[str, Path]) -> str:
    """
    Extract the simulation step of a path files
    :param raw_file: Path to the csv file
    :return: name of the step
    """
    raw_file = Path(raw_file)
    step_folder = list(Path(raw_file).parents)[0].name
    step_folder = str(step_folder)

    return step_folder


def extract_the_simulation_folder(file: Union[str, Path]) -> str:
    """
    Extract the simulation folder of a path
    :param file: Path or str of the path to decrypt
    :return: name of the simulation folder
    """
    file = Path(file)
    return str(list(file.parents)[2].name)


def remove_baseplate_from_list(files: List[Path]) -> List[Path]:
    """
    Remove the baseplate file from a list of files as we are not using it
    :param files: list of path
    :return: the same list but without the baseplate!
    """
    to_remove = list()
    for id_, file in enumerate(files):
        if "baseplate" in file.stem:
            to_remove.append(id_)

    to_remove.reverse()

    for id_ in to_remove:
        files.pop(id_)

    return files

def extract_simulation_folder(raw_files: List[Path]) -> List[str]:
    """
    Extract the simulation folders (not step folders) from the raw file list
    :param raw_files: List of all files
    :return: List of all the folder names (one per simulation)
    """
    simulation_folders = list()
    raw_files = remove_baseplate_from_list(raw_files)

    for file in raw_files:
        #: Folder name just bellow raw
        if file.name == "raw":
            continue
        simulation_folders.append(extract_the_simulation_folder(file))
    simulation_folders = list(dict.fromkeys(simulation_folders))

    return simulation_folders
