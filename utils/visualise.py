import torch
import pyvista as pv
from pathlib import Path
import numpy as np
from pytorch_lightning import LightningModule
from torch.utils.data import Subset
from torch_geometric.data import Data
import logging
from tqdm import tqdm
import shutil
from typing import Union


def display_dataset(model: LightningModule, dataset: Subset, configuration, folder_name: str = "display_dataset"):
    """  Generate vtk representation of the given dataset.

    Note: no loss are calculated.
    Note:0 all files in the given dataloader folder are processed.

    Parameters
    ----------
    model : LightningModule
        The neural network that should be used to generate the prediction.
    dataset: Subset
        Dataset used to test the neural network.
    configuration
        wandb configuration object. Used to scale the displayed values.
    folder_name: str
        Name of the folder where to save the vtk file.

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

        save_folder = Path(files[0]).parents[1] / folder_name

        if save_folder.exists():
            shutil.rmtree(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

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
            file_save = (save_folder / Path(file).stem).with_suffix(".vtk")
            create_vtk(batch, data, file_save)

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


def read_pt_batch_results(file: Union[str, Path], configuration):
    """ Read pt file created by the test step of the model.
    Load the saved batch results as pt file and created there vtk representation for analysis.

    For ease of coding, its expected that the graph.y contain graph.y and graph.y_hat stacked together.

    Example
    -------
    If the batch results of AMGNN were saved in the pt file "data/light/test_output/0.pt", then it's possible to
    generate a vtk file for each graph in the saved batch::

        from utils.config import read_config
        configuration = read_config(Path("configs"))
        read_pt_batch_results(r"data/light/test_output/0.pt", configuration)

    Parameters
    ----------
    file: str or Path
        Where the pt file to read is located. This pt file must be a saved batch.
        This must be the complete path to the pt file.

    configuration
        wandb configuration object. Used to scale the displayed values.

    Returns
    -------
    None
    """
    file = Path(file)
    batch = torch.load(file)

    # For every graph in the batch, we generate a visualisation:
    for graph_id in range(batch.num_graphs):
        graph = batch[graph_id].to("cpu")
        name = graph.file_name
        # graph.y is expected to be of shape [n, 8] as it's the stack of y and y_hat vector
        if graph.y.shape[1] == 8:
            y, y_hat = torch.split(graph.y, 4, dim=1) # Un split the saved y and y_hat features
        else:
            y = graph.y
            y_hat = None

        # Process the data to be displayed
        y_temperature = y[:, 0] * configuration["scaling_temperature"]
        y_disp_x = y[:, 1] * configuration["scaling_size"]
        y_disp_y = y[:, 2] * configuration["scaling_size"]
        y_disp_z = y[:, 3] * configuration["scaling_size"]
        y_displacement_vectors = np.vstack((y_disp_x, y_disp_y, y_disp_z)).T

        x_laser_speed = graph.x[:, 0] * configuration["scaling_speed"]
        x_laser_power = graph.x[:, 1] * configuration["scaling_power"]
        x_layer_thickness = graph.x[:, 2]
        x_time_step = graph.x[:, 3]
        x_type = np.argmax(graph.x[:, 4:7], axis=1)
        x_past_temp = graph.x[:, 7] * configuration["scaling_temperature"]
        x_past_x_displacement = graph.x[:, 8] * configuration["scaling_size"]
        x_past_y_displacement = graph.x[:, 9] * configuration["scaling_size"]
        x_past_z_displacement = graph.x[:, 10] * configuration["scaling_size"]


        # If we have some prediction
        if not y_hat is None:
            y_temperature_hat = y_hat[:, 0] * configuration["scaling_temperature"]
            y_disp_x_hat = y_hat[:, 1] * configuration["scaling_size"]
            y_disp_y_hat = y_hat[:, 2] * configuration["scaling_size"]
            y_disp_z_hat = y_hat[:, 3] * configuration["scaling_size"]
            y_hat_displacement_vectors = np.vstack((y_disp_x_hat, y_disp_y_hat, y_disp_z_hat)).T

            error_temperature = y_temperature - y_temperature_hat
            error_deformation = y_displacement_vectors - y_hat_displacement_vectors

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
                "Predicted displacement vector": y_hat_displacement_vectors,
                "Error temperature": error_temperature,
                "Error deformatione" : error_deformation
            }
        else:
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
                "Label displacement vector": y_displacement_vectors,
                }

        # Save the vtk file
        file_save = (file.parent / name).with_suffix(".vtk")
        create_vtk(graph, data, file_save)


if __name__ == "__main__":
    from utils.config import read_config
    configuration = read_config(Path("configs"))
    read_pt_batch_results(r"data/light/test_output/0.pt", configuration)
