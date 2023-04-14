import torch
from scipy.spatial import KDTree


@torch.no_grad()
def align_features(actual_features: torch.Tensor, past_features: torch.Tensor, config: dict) -> \
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align the actual features and the past features.

    Take 2 tensors representing the features and coordinates of two arc files (one at time t and one at time -1).
    Remove the deformation of the coordinates and try to match the features of each point to merge them.
    Use a KDTree to match the coordinates.
    :param past_features:
    :param actual_features:
    :param config: dictionary containing the scalling parameters
    :return: actual_coor_no_def, X, Y the actual input and expected output of the neural network.

    Parameters
    ----------
    actual_features: Tensor
        Tensor representation of the actual features.
    past_features: Tensor
        Tensor representation of the past features.
    config: dict
        configuration dictionary used to normalise and unnormalise data and set default value.

    Returns
    -------
    Tensor, Tensor, Tensor
    """

    # Create the true X and Y tensors
    X = torch.zeros((actual_features.shape[0], 12))
    Y = torch.zeros((actual_features.shape[0], 4))

    past_coor_no_def = past_features[:, :3]
    actual_coor_no_def = actual_features[:, :3].detach().clone()
    kd = KDTree(past_coor_no_def * config["scaling_size"])

    for i, coor in enumerate(actual_coor_no_def):

        distance, id_point = kd.query(coor * config["scaling_size"],
                                      distance_upper_bound=float(config["distance_upper_bound"]))

        if distance == float("inf"):
            # If no matching point are returned, set default value.
            # With coordinate as input
            #             good_features = torch.concat((actual_coor_no_def[i], actual_features[i, 3:-4],
            #                                         torch.tensor([wandb.config.default_powder_temperature, 0, 0, 0])))
            good_features = torch.concat((actual_features[i, 3:-4], torch.tensor([
                float(config["default_powder_temperature"]) / float(config["scaling_temperature"]), 0, 0, 0])))
            good_target = actual_features[i, -4:]
        else:
            #  past features have the following data:
            #       past_coordinates, x_past_temp, x_past_xdis, x_past_ydis, x_past_zdis
            # actual features have the following data:
            # coordinates, x_laser_speed, x_laser_power, x_layer_thickness, x_time_step_length,
            # x_type, y_temp, y_xdis, y_ydis,y_zdis
            # If there is a match, add feature to actual features

            # remove the actual deformatation/ temperature to add the past one
            # With the coordinate as input
            # good_features = torch.concat((actual_coor_no_def[i], \
            # actual_features[i, 3:-4], past_features[id_point, 3:]))
            # without the coordinate as input
            good_features = torch.concat((actual_features[i, 3:-4], past_features[id_point, 3:]))
            good_target = actual_features[i, -4:]
        X[i] = good_features
        Y[i] = good_target

    return actual_coor_no_def, X, Y


@torch.no_grad()
def align_x_and_y(x1: torch.Tensor, y1: torch.Tensor, x2: torch.Tensor, config: dict) -> torch.Tensor:
    """Align an graph x tensor with the y AI result from a previous graph.
    Return the new graph x tensor with AI results.

    Parameters
    ----------
    x1: torch.Tensor, Features tensor of the previous graph.
    y1: torch.Tensor, Results tensor of the AI.
    x2: torch.Tensor, Features tensor of the next graph
    config: dict, containing the scaling size to scale up the point coordinates for the alignment.

    Returns
    -------
    x2y: torch.Tensor, Features tensor of the next graph with AI previous results.
    """
    # Initialise the return tensor
    x2y = x2.clone()

    # Default scaled temperature
    def_temperature = float(config["default_powder_temperature"]) / float(config["scaling_temperature"])

    # Create a tensor with a scaled displacement of 0.5 and a scaled temperature of def_temperature
    x2y[:, 8:12] = torch.tensor([[def_temperature, 0.5, 0.5, 0.5]] * x2y.shape[0])

    # Unscale coordinate
    x1_coor = x1[:, -3:] * config["scaling_size"]
    x2_coor = x2[:, -3:] * config["scaling_size"]

    # Create the KD tree for alignment
    kd = KDTree(x2_coor)

    # For all points in x1, look for the closest in x2
    for i, coor in enumerate(x1_coor):
        distance, id_point = kd.query(coor * config["scaling_size"],
                                      distance_upper_bound=float(config["distance_upper_bound"]))
        if distance == float("inf"):
            # No matching point have been found (no point in a radius of distance_upper_bound)
            pass
        else:
            # The point i in x1 correspond to the point id_point in x2
            x2y[id_point] = y1[i].clone()

    return x2y
