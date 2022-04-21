import torch
from scipy.spatial import KDTree


@torch.no_grad()
def align_features(actual_features: torch.Tensor, past_features: torch.Tensor, config: dict) -> \
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Take 2 tensors representing the features and coordinates of two arc files.
    Remove the deformation of the coordinates and try to match the features of each point to merge them.
    Use a KDTree to match the coordinates.
    :param past_features:
    :param actual_features:
    :param config: dictionary containing the scalling parameters
    :return: actual_coor_no_def, X, Y the actual input and expected output of the neural network.
    """

    # Create the true X and Y tensors
    X = torch.zeros((actual_features.shape[0], 11), names=["voxels", "features"])
    Y = torch.zeros((actual_features.shape[0], 4), names=["voxels", "target"])

    past_coor_no_def = past_features[:, :3] - past_features[:, 4:7]
    actual_coor_no_def = torch.tensor(actual_features[:, :3] - actual_features[:, 11:14])
    kd = KDTree(past_coor_no_def)

    for i, coor in enumerate(actual_coor_no_def):

        distance, id_point = kd.query(coor, distance_upper_bound=float(config["distance_upper_bound"]))

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
