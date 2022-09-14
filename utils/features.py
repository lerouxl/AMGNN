import numpy as np
import wandb
from torch import Tensor
from simufact_arc_reader.ARC_CSV import Arc_reader
from typing import Tuple
import torch
import torch.nn.functional as F
from utils.edges_creation import create_edge_list_and_length
from numba import njit
from sklearn import preprocessing
from utils.align_features import align_features


@torch.no_grad()
def arc_features_extraction(arc: Arc_reader, past_arc: Arc_reader, config: dict) -> \
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Extract the features from an ARC files and return there values as torch tensors.
    All features are scaled using the config data
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
    :param past_arc: Arc of the previous simulation step
    :param neighbour_k: Max number of edges per points
    :param distance_upper_bound: Max distance between point to have an edge
    :param config: dictionary with all the configuration variables.
    :return:
    """
    neighbour_k = int(config["neighbour_k"])
    distance_upper_bound = float(config["distance_upper_bound"])

    # Extract the points of the point cloud
    coordinates = torch.tensor(arc.coordinate, dtype=torch.float)

    # Extract labels
    y_temp = torch.tensor(arc.data.TEMPTURE, dtype=torch.float)  # Shape [n_points]
    y_xdis = torch.tensor(arc.data.XDIS, dtype=torch.float)  # Shape [n_points]
    y_ydis = torch.tensor(arc.data.YDIS, dtype=torch.float)  # Shape [n_points]
    y_zdis = torch.tensor(arc.data.ZDIS, dtype=torch.float)  # Shape [n_points]

    # From meter to mm
    y_xdis = 100 * y_xdis
    y_ydis = 100 * y_ydis
    y_zdis = 100 * y_zdis

    # Shape [4, n_points]

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
    past_coordinates = torch.tensor(past_arc.coordinate, dtype=torch.float)
    x_past_temp = torch.tensor(past_arc.data.TEMPTURE, dtype=torch.float)  # Shape [n_points]
    x_past_xdis = torch.tensor(past_arc.data.XDIS, dtype=torch.float)  # Shape [n_points]
    x_past_ydis = torch.tensor(past_arc.data.YDIS, dtype=torch.float)  # Shape [n_points]
    x_past_zdis = torch.tensor(past_arc.data.ZDIS, dtype=torch.float)  # Shape [n_points]

    x_laser_speed = torch.full(y_temp.shape, arc.metaparameters.speed_m_s, dtype=torch.float)
    x_laser_power = torch.full(y_temp.shape, arc.metaparameters.power_W, dtype=torch.float)
    x_layer_thickness = torch.full(y_temp.shape, arc.metaparameters.layerThickness_m * 1e4, dtype=torch.float)
    x_time_step = torch.full(y_temp.shape, arc.metaparameters.time_steps_s, dtype=torch.float)
    x_time_step_length = torch.full(y_temp.shape, arc.metaparameters.time_steps_length_s, dtype=torch.float)
    # x = torch.stack([x_laser_speed,x_laser_power,x_layer_thickness,x_time_step,x_time_step_length, x_type])

    # Scale printing parameters
    x_laser_power = x_laser_power / float(config["scaling_power"])  # the laser power will be between [0,1]
    x_laser_speed = x_laser_speed / float(config["scaling_speed"])

    # scale max temperature
    x_past_temp = x_past_temp / float(config["scaling_temperature"])
    y_temp = y_temp / float(config["scaling_temperature"])

    # From meter to mm
    x_past_xdis = 100 * x_past_xdis
    x_past_ydis = 100 * x_past_ydis
    x_past_zdis = 100 * x_past_zdis
    coordinates = 100 * coordinates
    past_coordinates = 100 * past_coordinates

    # scale
    y_xdis /= float(config["scaling_size"])
    y_ydis /= float(config["scaling_size"])
    y_zdis /= float(config["scaling_size"])
    x_past_xdis /= float(config["scaling_size"])
    x_past_ydis /= float(config["scaling_size"])
    x_past_zdis /= float(config["scaling_size"])
    past_coordinates /= float(config["scaling_size"])
    coordinates /= float(config["scaling_size"])

    actual_features = torch.concat((coordinates, x_laser_speed.unsqueeze(1), x_laser_power.unsqueeze(1), \
                                    x_layer_thickness.unsqueeze(1), x_time_step_length.unsqueeze(1), \
                                    x_type, y_temp.unsqueeze(1), y_xdis.unsqueeze(1), y_ydis.unsqueeze(1), \
                                    y_zdis.unsqueeze(1)), axis=1)
    past_features = torch.concat((past_coordinates, x_past_temp.unsqueeze(1), x_past_xdis.unsqueeze(1), \
                                  x_past_ydis.unsqueeze(1), x_past_zdis.unsqueeze(1)), axis=1)

    coordinates, X, Y = align_features(actual_features, past_features, config)

    # load the edges
    part_edge_index = torch.tensor(arc.edge_index, dtype=torch.long)

    return coordinates, part_edge_index, X, Y
