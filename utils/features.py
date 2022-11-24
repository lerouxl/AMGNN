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
from model.amgnn import check_tensors


@torch.no_grad()
def arc_features_extraction(arc: Arc_reader, past_arc: Arc_reader, config: dict) -> \
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Extract the features from an ARC files and return there values as torch tensors.

    All features are scaled using the config data
    The extracted features are:
        - coordinates
        - edges
        - edges lengt
        - X: 22 features for each nodes used for predict Y
            - Laser speed (m/s scaled with `scaling_speed`)
            - Laser power (W scaled with `scaling_power`)
            - Layer thickness (um/100)
            - Time step length (s  scaled by `100 000`)
            - Time step (s scaled with `scalling_time`)
            - Type: One hot vector for ["baseplate", "part", "supports"]
            - Past temperature (in Celsius scaled with `scaling_temperature`)
            - Past displacement X (in `mm` scaled with `scaling_deformation`)
            - Past displacement Y (in `mm` scaled with `scaling_deformation`)
            - Past displacement Z (in `mm` scaled with `scaling_deformation`)
            - Process features (number of printed voxel layer scaled with `max_number_of_layer`)
            - Process category: One hot vector for [AM_Layer, process, Postcooling, Powderromval, Unclamping, Cooling-1]
            - Coordinate X (in `mm` scaled with scaling_size)
            - Coordinate Y (in `mm` scaled with scaling_size)
            - Coordinate Z (in `mm` scaled with scaling_size)
        - Y (target):
            - TEMPTURE (in Celsius scaled with `scaling_temperature`)
            - XDIS (in `mm` scaled with `scaling_deformation`)
            - YDIS (in `mm` scaled with `scaling_deformation`)
            - ZDIS (in `mm` scaled with `scaling_deformation`)

    Parameters
    ----------
    arc : Arc_reader
        Merged arcs files.
    past_arc : Arc_reader
        Arc of the previous simulation step.
    config: dict
        Dictionary with all the configuration variables.

    Returns
    -------
    coordinates: torch.Tensor
        Tensor of the nodes coordinates of shape [n, 3] with n the number of nodes.
    part_edge_index: torch.Tensor
        Tensor listing all edges of the mesh. Of shape [2,e] with e the number of edges.
    X: torch.Tensor
        Input tensor for the deep learning model of shape [n,18] with n the number of nodes.
    Y: torch.Tensor
        Label tensor for the deep learning model of shape [n,4] with n the number of nodes.
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

    # Extract simulation process steps category and features
    process_category = torch.tensor(arc.data.process_category, dtype=torch.float)
    process_features = torch.tensor(arc.data.process_features, dtype=torch.float)

    # Previous step results
    past_coordinates = torch.tensor(past_arc.coordinate, dtype=torch.float)
    x_past_temp = torch.tensor(past_arc.data.TEMPTURE, dtype=torch.float)  # Shape [n_points]
    x_past_xdis = torch.tensor(past_arc.data.XDIS, dtype=torch.float)  # Shape [n_points]
    x_past_ydis = torch.tensor(past_arc.data.YDIS, dtype=torch.float)  # Shape [n_points]
    x_past_zdis = torch.tensor(past_arc.data.ZDIS, dtype=torch.float)  # Shape [n_points]

    x_laser_speed = torch.full(y_temp.shape, arc.metaparameters.speed_m_s, dtype=torch.float)
    x_laser_power = torch.full(y_temp.shape, arc.metaparameters.power_W, dtype=torch.float)
    x_layer_thickness = torch.full(y_temp.shape, arc.metaparameters.layerThickness_m, dtype=torch.float)
    x_time_step = torch.full(y_temp.shape, arc.metaparameters.time_steps_s, dtype=torch.float)
    x_time_step_length = torch.full(y_temp.shape, arc.metaparameters.time_steps_length_s, dtype=torch.float)
    # x = torch.stack([x_laser_speed,x_laser_power,x_layer_thickness,x_time_step,x_time_step_length, x_type])

    # Scale printing parameters
    x_laser_power = x_laser_power / float(config["scaling_power"])  # the laser power will be between [0,1]
    x_laser_speed = x_laser_speed / float(config["scaling_speed"])

    # Scale time step length
    x_time_step_length = x_time_step_length / 100_000

    #scale time
    x_time_step = x_time_step / float(config["scalling_time"])

    # scale max temperature
    x_past_temp = x_past_temp / float(config["scaling_temperature"])
    y_temp = y_temp / float(config["scaling_temperature"])


    # scale
    y_xdis = y_xdis / float(config["scaling_deformation"])
    y_ydis = y_ydis / float(config["scaling_deformation"])
    y_zdis = y_zdis / float(config["scaling_deformation"])
    x_past_xdis = x_past_xdis / float(config["scaling_deformation"])
    x_past_ydis = x_past_ydis / float(config["scaling_deformation"])
    x_past_zdis = x_past_zdis / float(config["scaling_deformation"])
    past_coordinates /= float(config["scaling_size"])
    coordinates /= float(config["scaling_size"])
    process_features /= float(config["max_number_of_layer"])

    actual_features = torch.concat((coordinates, x_laser_speed.unsqueeze(1), x_laser_power.unsqueeze(1),
                                    x_layer_thickness.unsqueeze(1), x_time_step_length.unsqueeze(1),
                                    x_time_step.unsqueeze(1),x_type, y_temp.unsqueeze(1),
                                    y_xdis.unsqueeze(1), y_ydis.unsqueeze(1), y_zdis.unsqueeze(1)), axis=1)
    past_features = torch.concat((past_coordinates, x_past_temp.unsqueeze(1), x_past_xdis.unsqueeze(1),
                                  x_past_ydis.unsqueeze(1), x_past_zdis.unsqueeze(1)), axis=1)

    coordinates, X, Y = align_features(actual_features, past_features, config)

    # load the edges
    part_edge_index = torch.tensor(arc.edge_index, dtype=torch.long)

    # Add the subprocess features to the nodes
    X = torch.cat([X, process_features.unsqueeze(1)], 1) # [n, 13]
    X = torch.cat([X, process_category],1) # [n, 19]
    X = torch.cat([X, coordinates],1) # [n, 22]

    # Check that no infinite value, NaN value or values superior to 1 are in the input and target tensors
    check_tensors(X)
    check_tensors(Y)
    return coordinates, part_edge_index, X, Y

