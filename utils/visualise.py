import torch
import pyvista as pv
from pathlib import Path
import numpy as np
from model.amgnn import AMGNNmodel
from torch.utils.data import Subset
from torch_geometric.data import Data
import logging
from tqdm import tqdm


def display_dataset(model: AMGNNmodel, dataset: Subset, configuration):
    """  Generate vtk representation of the given dataset.

    Note: no loss are calculated.
    Note:0 all files in the given dataloader folder are processed.

    Parameters
    ----------
    model : AMGNNmodel
        The neural network that should be used to generate the prediction.
    dataset: Subset
        Dataset used to test the neural network.
    configuration
        wandb configuration object. Used to scale the displayed values.

    Returns
    -------

    """

    log = logging.getLogger(__name__)
    log.info("Start the display dataset function")
    with torch.no_grad():
        model.eval()

        # Iterate on the pt files
        files = dataset.dataset.processed_paths
        files = list(files)

        for file in tqdm(files):

            file = Path(file)
            log.debug("Create vtk file of {file}")

            batch = torch.load(file)
            y_hat = model.network(batch)

            # Process the data to be displayed
            y_temperature = batch.y[:, 0] * configuration["scaling_temperature"]
            y_disp_x = batch.y[:, 1] * configuration["scaling_size"]
            y_disp_y = batch.y[:, 2] * configuration["scaling_size"]
            y_disp_z = batch.y[:, 3] * configuration["scaling_size"]
            y_displacement_vectors = np.vstack((y_disp_x, y_disp_y, y_disp_z)).T

            y_temperature_hat = y_hat[:, 0] * configuration["scaling_temperature"]
            y_disp_x_hat = y_hat[:, 1] * configuration["scaling_size"]
            y_disp_y_hat = y_hat[:, 2] * configuration["scaling_size"]
            y_disp_z_hat = y_hat[:, 3] * configuration["scaling_size"]
            y_hat_displacement_vectors = np.vstack((y_disp_x_hat, y_disp_y_hat, y_disp_z_hat)).T

            x_laser_speed = batch.x[:, 0] * configuration["scaling_speed"]
            x_laser_power = batch.x[:, 1] * configuration["scaling_power"]
            x_layer_thickness = batch.x[:, 2]
            x_time_step = batch.x[:, 3]
            x_type = np.argmax(batch.x[:, 4:7], axis=1)
            x_past_temp = batch.x[:, 7] * configuration["scaling_temperature"]
            x_past_x_displacement = batch.x[:, 8] * configuration["scaling_size"]
            x_past_y_displacement = batch.x[:, 9] * configuration["scaling_size"]
            x_past_z_displacement = batch.x[:, 10] * configuration["scaling_size"]

            # Create a dictionary representation
            data = {
                "laser speed": x_laser_speed,
                "laser power": x_laser_power,
                "layer thickness": x_layer_thickness,
                "time step": x_time_step,
                "type": x_type,
                "past temperature": x_past_temp,
                "past displacement X": x_past_x_displacement,
                "past displacement Y": x_past_y_displacement,
                "past displacement Z": x_past_z_displacement,
                "Label temperature": y_temperature,
                "Predicted temperature": y_temperature_hat,
                "Label displacement vector": y_displacement_vectors,
                "Predicted displacement vector": y_hat_displacement_vectors
            }

            # Save the vtk file
            create_vtk(batch, data, file.with_suffix(".vtk"))

    log.info("End of the display dataset function")


def create_vtk(graph: Data, nodes_features: dict, save_path: Path) -> None:
    """ Create a vtk files with the given features.

    Giving a graph, a vtk file is created with for nodes features the ones given in the dictionary nodes_features.
    The file is then saved in save_path. The generated vtk file can been seen with Paraview as a wireframe.

    Parameters
    ----------
    graph : Data
        Torch geometric Data class containing one graph.
    nodes_features: dict
        Dictionary containing the name and numpy values of all nodes features.
    save_path: Path
        Path where to save the vtk file.

    Returns
    -------

    """
    # Enforce save_path type if an str is given
    save_path = Path(save_path)
    # For the output file to be a vtk file in case the pt file name was given
    if save_path.suffixes[0] != ".vtk":
        save_path = save_path.with_suffix(".vtk")

    # Graph creation
    edges = graph.edge_index.T.numpy()
    nodes = graph.pos.numpy()

    # Pad the edges list
    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T

    # Create the pyvista mesh (can been see with the wireframe)
    mesh = pv.PolyData(nodes, edges_w_padding)

    # Loop on all nodes_features dictionary keys and add them to the wireframe.
    for key in nodes_features.keys():
        mesh.point_data[str(key)] = nodes_features[key]

    # Save the mesh visualisation
    mesh.save(save_path)

