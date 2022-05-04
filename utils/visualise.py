import torch
import pyvista as pv
from pathlib import Path
import numpy as np
from model.amgnn import AMGNNmodel
from torch.utils.data import Subset


def display_dataset(model: AMGNNmodel, dataset: Subset, configuration):
    """
    Generate vtk representation of the dataset.
    Note, no loss are calculated.
    Note, all files in the dataloader folder are processed.
    :param model: Neural network used to generate the prediction
    :param dataset: Dataset used to test the neural network
    :param configuration: wandb configuration object. Used to scale the displayed values.
    :return:
    """
    with torch.no_grad():
        model.eval()

        # Iterate on the pt files
        files = dataset.dataset.processed_paths
        for file in files:
            batch = torch.load(file)
            file = Path(file)

            y_hat = model.network(batch)
            display_x_y_y_hat_data(pos=batch.pos.numpy(), x=batch.x.numpy(),
                                   y=batch.y.numpy(), local_path=file, y_hat=y_hat.numpy(),
                                   config=configuration)


def display_x_y_y_hat_data(pos, x: np.ndarray, y: np.ndarray, local_path: Path, y_hat: np.ndarray = None,
                           display_features=True, config=None):
    """
    Display the results of the neural network using :
    :param pos: the 3D position of each voxels center
    :param x: The features of each voxels
    :param y: The target of each voxels
    :param local_path: Path to the loaded file, used to save the output vtk in the same folder.
    :param y_hat: The network output
    :param display_features: If the input features must be displayed
    :param config: Values scaling configuration (Note, features are not scalled)
    :return:
    """

    if config is None:
        scaling_temperature = 1
        scaling_size = 1
        scaling_speed = 1
        scaling_power = 1
        scaling_size = 1
    else:
        scaling_temperature = config["scaling_temperature"]
        scaling_size = config["scaling_size"]
        scaling_speed = config["scaling_speed"]
        scaling_power = config["scaling_power"]
        scaling_size = config["scaling_size"]

    cloud_point = pv.PolyData(pos * scaling_size)

    if display_features:
        features_list = ["laser speed", "laser power", "layer thickness", "time step duration", "baseplate", "part",
                         "supports", "scaled past temperature"]
        #"x past displacement", "y past displacement", "z past displacement"
        for i, name in enumerate(features_list):
            value = x[:, i]
            if name == "laser power":
                value *= scaling_power
            elif name == "laser speed":
                value *= scaling_speed
            elif name == "scaled past temperature":
                value *= scaling_temperature

            cloud_point.point_data["Feature " + name] = value

        # Generate the past displacement vector
        value = x[:, -4:-1] # " past displacement, y past displacement, z past displacement
        value *= scaling_size
        cloud_point.point_data["Feature past displacement (mm)"] = value

    # Add target
    cloud_point.point_data["Target temperature (C)"] = y[:, 0] * scaling_temperature
    cloud_point.point_data["Target displacement (mm)"] = y[:, 1:-1] * scaling_size

    if y_hat is not None:
        # Add prediction
        cloud_point.point_data["Predicted temperature (C)"] = y_hat[:, 0] * scaling_temperature
        cloud_point.point_data["Predicted displacement (mm)"] = y_hat[:, 1:-1] * scaling_size

        # Add error
        cloud_point.point_data["Error displacement (y - y_hat) (mm)"] = (y[:, 1:-1] - y_hat[:, 1:-1]) * scaling_size
        cloud_point.point_data["Error temperature (y - y_hat) (C)"] = (y[:, 0] - y_hat[:, 0]) * scaling_temperature

    cloud_point.save(local_path.with_suffix(".vtk"))
