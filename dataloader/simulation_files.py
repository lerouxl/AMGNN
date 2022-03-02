from typing import Union, List
from pathlib import Path


def organise_files(raw_files: Union[List[Path], List[str]]) -> List:
    """
    Take the list of raw files and join the file of the same simulation step together in a list of list.
    :param raw_files: List of all files, can be str of Path data
    :return:
    """

    organised_list = raw_files
    return organised_list


def extract_simulation_folder(raw_files: Union[List[Path], List[str]]) -> List[str]:
    """
    Extract the simulation folders (not step folders) from the raw file list
    :param raw_files: List of all files
    :return: List of all the folder names (one per simulation)
    """
    simulation_folders = list()

    for file in raw_files:
        #: Folder name just bellow raw
        simulation_folders.append(list(Path(file).parents)[3].name)
    simulation_folders = list(dict.fromkeys(simulation_folders))

    return simulation_folders
