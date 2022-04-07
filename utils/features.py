import numpy as np
from torch import Tensor
from Simufact_ARC_reader.ARC_CSV import Arc_reader
from typing import Tuple
import torch
import torch.nn.functional as F
from utils.edges_creation import create_edge_list_and_length
from numba import njit
from sklearn import preprocessing


def arc_features_extraction(arc: Arc_reader, past_arc: Arc_reader, neighbour_k: int, distance_upper_bound: float) -> \
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Extract the features from an ARC files and return there values as torch tensors.
    The extracted features are:
        - coordinates
        - edges
        - edges length
        - x:
            - Type (support, part) (one hot)
            - Laser Power
            - Laser Speed
            - Layer height
            - Layer processing time
            - Past points temperature
            - Past point X displacement
            - Past point Y displacement
            - Past point Z displacement
        - Y (target): "XDIS", "YDIS", "ZDIS", "TEMPTURE"

    :param arc: Merged arcs files
    :param neighbour_k: Max number of edges per points
    :param distance_upper_bound: Max distance between point to have an edge
    :return:
    """
    # Extract the points of the point cloud
    coordinates = torch.tensor(arc.coordinate, dtype=torch.float)

    # Create the edges and compute there size
    part_edge_index, length = create_edge_list_and_length(neighbour_k=neighbour_k,
                                                          distance_upper_bound=distance_upper_bound,
                                                          coordinates=arc.coordinate)
    part_edge_index = torch.tensor(part_edge_index, dtype=torch.int).t().contiguous()
    length = torch.tensor(length, dtype=torch.float)

    # Extract labels
    y_temp = torch.tensor(arc.data.TEMPTURE, dtype=torch.float)  # Shape [n_points]
    y_xdis = torch.tensor(arc.data.XDIS, dtype=torch.float)  # Shape [n_points]
    y_ydis = torch.tensor(arc.data.YDIS, dtype=torch.float)  # Shape [n_points]
    y_zdis = torch.tensor(arc.data.ZDIS, dtype=torch.float)  # Shape [n_points]

    y = torch.stack([y_temp, y_xdis, y_ydis, y_zdis])  # Shape [4, n_points]

    # Extract x features
    #   - Type (support, part) An one hot vector of the type of part.
    onehot = preprocessing.OneHotEncoder(categories=[["baseplate", "part", "supports"]])
    onehot.fit(arc.points_types.reshape(-1, 1))
    x_type = onehot.transform(arc.points_types.reshape(-1, 1)).toarray()
    x_type = torch.tensor(x_type, dtype=torch.int32)

    #   - Laser Power
    # See note from 2022 01 14
    #   - Laser Speed
    #   - Layer height
    #   - Layer processing time
    #   - Past points temperature
    #   - Past point X displacement
    #   - Past point Y displacement
    #   - Past point Z displacement

    # Previous step results
    # TODO: Alligner les anciens points avec les nouveaux.
    x_past_temp = torch.tensor(past_arc.data.TEMPTURE, dtype=torch.float)  # Shape [n_points]
    x_past_xdis = torch.tensor(past_arc.data.XDIS, dtype=torch.float)  # Shape [n_points]
    x_past_ydis = torch.tensor(past_arc.data.YDIS, dtype=torch.float)  # Shape [n_points]
    x_past_zdis = torch.tensor(past_arc.data.ZDIS, dtype=torch.float)  # Shape [n_points]

    x_laser_speed = torch.full(y_temp.shape, arc.metaparameters.speed_m_s, dtype=torch.float)
    x_laser_power = torch.full(y_temp.shape, arc.metaparameters.power_W, dtype=torch.float)
    x_layer_thickness = torch.full(y_temp.shape, arc.metaparameters.layerThickness_m * 1e4, dtype=torch.float)
    x_time_step = torch.full(y_temp.shape, arc.metaparameters.time_steps_s, dtype=torch.float)
    x_time_step_length = torch.full(y_temp.shape, arc.metaparameters.time_steps_lenght_s, dtype=torch.float)
    x = torch.stack([x_laser_speed,x_laser_power,x_layer_thickness,x_time_step,x_time_step_length, x_type])

    return coordinates, part_edge_index, length, x, y
