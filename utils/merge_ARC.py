"""
Take multiple ARC files, merge them and create a mesh.
"""
from typing import List
import numpy as np
import copy
from simufact_arc_reader.ARC_CSV import Arc_reader


def merge_arc(arc_files: List[Arc_reader]) -> Arc_reader:
    """
    Merge ARC files and reconstruct the data matrix.
    :param arc_files: list of arc files.
    :return: merged ARCs files as an Arc_reader object
    """
    merged = copy.deepcopy(arc_files[0])
    merged.name = "merged_file"
    #  Create a vector containing the type of the coordinate.
    # Max text lenght = 9 (S9) as baseplate = 9 characters
    merged.points_types = np.full_like(merged.data.TEMPTURE, str(merged.arc_type), dtype="S9")
    merged.arc_type = "merged"

    # Transform the metaparameters in array
    merged = metaparameters_to_array(merged)

    # TODO: Speed up with Numba?
    for arc_file in arc_files[1:]:
        arc_file = metaparameters_to_array(arc_file)
        # Add the new coordinates
        merged.coordinate = np.append(merged.coordinate, arc_file.coordinate, 0)
        # Add the point types (part, supports or baseplate)
        merged.points_types = np.append(merged.points_types,
                                        np.full_like(arc_file.data.TEMPTURE, str(arc_file.arc_type), dtype="S9")
                                        )
        # Add the "new" edges
        # The edges index is shifted so the lowest new edges index is equal to the first new coordinate

        merged.edge_index[0].extend([ei + len(merged.connectivity) for ei in arc_file.edge_index[0]])
        merged.edge_index[1].extend([ei + len(merged.connectivity) for ei in arc_file.edge_index[1]])


        data_attributes = [attr for attr in dir(merged.data) if not str(attr).startswith("__")]
        for data_attribute in data_attributes:
            merged_data = getattr(merged.data, data_attribute)
            new_data = getattr(arc_file.data, data_attribute)
            merged_data = np.append(merged_data, new_data, 0)
            setattr(merged.data, data_attribute, merged_data)

        # merge meta data (laser power, speed, layer thickness, times, times spend)
        metadata_attributes = [attr for attr in dir(merged.metaparameters) if not str(attr).startswith("__")]
        for metadata_attribute in metadata_attributes:
            merged_data = getattr(merged.metaparameters, metadata_attribute)
            new_data = getattr(arc_file.metaparameters, metadata_attribute)
            merged_data = np.append(merged_data, new_data, 0)
            setattr(merged.metaparameters, metadata_attribute, merged_data)

    # Create an edge between nodes that are at the same position (ex: supports to part link)
    values, indices, counts = np.unique(merged.coordinate, return_inverse=True, return_counts=True, axis=0)

    new_edges = []
    for c in (counts > 1).nonzero()[0]:
        new_edges.append((indices == c).nonzero()[0])

    for edge in new_edges:
        merged.add_edge(edge[0], edge[1])

    return merged

def metaparameters_to_array(arc: Arc_reader)-> Arc_reader:
    """
    Transform the metaparamters into an array of the size of the number of point.
    :param arc: The arc file
    :return: The same arc file the the metaparameters transformed into array
    """
    attributes = [attr for attr in dir(arc.metaparameters) if not str(attr).startswith("__")]
    for metadata_attribute in attributes:
        data = getattr(arc.metaparameters, metadata_attribute)
        data_array = np.full_like(arc.data.TEMPTURE, data, dtype=float)
        setattr(arc.metaparameters, metadata_attribute, data_array)

    return arc
