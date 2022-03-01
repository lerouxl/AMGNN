"""
Take multiple ARC files, merge them and create a mesh.
"""
from typing import List
import numpy as np
import copy
from Simufact_ARC_reader.ARC_CSV import Arc_reader
import wandb

def merge_ARC(arc_files: List ):
    """
    Merge ARC files and reconstruct the data matrix.
    :param arc_files: list of arc files.
    :return:
    """
    merged = copy.deepcopy(arc_files[0])
    merged.name = "merged_file"

    for arc_file in arc_files[1:]:
        # Add the new coordinates
        merged.coordinate = np.append(merged.coordinate, arc_file.coordinate,0)

        data_attributes = [attr for attr in dir(merged.data) if not str(attr).startswith("__")]
        for data_attribute in data_attributes:
            merged_data = getattr(merged.data, data_attribute)
            new_data = getattr(arc_file.data, data_attribute)
            merged_data = np.append(merged_data, new_data, 0)
            setattr(merged.data, data_attribute, merged_data)

    return merged
