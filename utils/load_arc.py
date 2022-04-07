from Simufact_ARC_reader.ARC_CSV import Arc_reader
from utils.merge_ARC import merge_arc
from typing import List, Union
from pathlib import Path, PurePath


def load_arcs_list(raw_paths: Union[List[Path], List[str]]) -> Arc_reader:
    """
    Take a list of Path, load those ARC files and merge them to make one ARC_reader object.

    :param raw_paths: List of the path to the ARC files
    :return: Arc_reader objects
    """
    step_arcs = list()
    for raw_path in raw_paths:
        # Read data from `raw_path`.
        raw_path = Path(raw_path)
        arc = load_arc(raw_path)
        step_arcs.append(arc)

    # Merge the listed files for this part and simulation step
    arc = merge_arc(step_arcs)
    return arc


def load_arc(raw_path: Union[Path, str]):
    """
    Load one ARC file and launch the processing.
    :param raw_path:
    :return:
    """
    if not isinstance(raw_path, PurePath):
        raw_path = Path(raw_path)

    file_name = raw_path.stem
    # Create an Arc_reader object to read the csv and create a graph
    arc = Arc_reader(name=file_name)
    # Read a csv dump
    arc.load_csv(raw_path)
    # Extract the point cloud coordinate
    arc.get_coordinate()
    # Add at each point all extract data
    arc.get_point_cloud_data(display=False)
    return arc
