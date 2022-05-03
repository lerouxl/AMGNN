import polyscope as ps
import numpy as np


def display_X_Y_Y_hat_data(pos, X, Y, Y_hat=None, display_features=False, config=None):
    """
    Display the results of the neural network using :
    :param pos: the 3D position of each voxels center
    :param X: The features of each voxels
    :param Y: The target of each voxels
    :param Y_hat: The network output
    :param display_features: If the input features must be displayed
    :param config: Values scaling configuration (Note, features are not scalled)
    :return:
    """
    ps.init()

    if config is None:
        scaling_temperature = 1
        scaling_size = 1
    else:
        scaling_temperature = float(config["scaling_temperature"])
        scaling_size = float(config["scaling_size"])

    cloud_point = ps.register_point_cloud("my points", pos * scaling_size)

    if display_features:
        features_list = ["laser", "speed", "laser power", "layer thickness", "time step duration", "x_type1", "x_type2",
                         "x_type3", "scaled past temperature", "x past displacement", "y past displacement",
                         "z past displacement"]
        for i, name in enumerate(features_list):
            cloud_point.add_scalar_quantity(name, X[:, i])

    # Add target
    cloud_point.add_scalar_quantity("Target Temperature", Y[:, 0] * scaling_temperature,
                                    vminmax=(0, scaling_temperature))
    cloud_point.add_vector_quantity("Target displacement", Y[:, 1:-1] * scaling_size)

    if Y_hat is not None:
        # Add prediction
        cloud_point.add_scalar_quantity("Predicted Temperature", Y_hat[:, 0] * scaling_temperature,
                                        vminmax=(0, scaling_temperature))
        cloud_point.add_vector_quantity("Predicted displacement", Y_hat[:, 1:-1] * scaling_size)

    ps.show()
