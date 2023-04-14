"""
Apply AMGNN to all simulation steps of a part using the previous prediction as input and
calculating the final reconstruction error.

This script work by:
1- Loading the AI
2- Loading the simulation .pt files
3- Simulate all layers using results from the past predicted layer
4- Evaluate the error
"""
from model.amgnn import AMGNNmodel
from dataloader.arc_dataset import ARCDataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from utils.align_features import align_x_and_y
from utils.config import read_config
from utils.shape_comparison import surface_reconstruction_error
from tqdm import tqdm
from pathlib import Path
from torch import Tensor
import torch


def load_model(model_path: str) -> AMGNNmodel:
    """Load the AI from a ckpt file.

    Parameters
    ----------
    model_path : str, path of the ckp file.

    Returns
    ------
    model: AMGNNmodel, the lightning configured model

    """
    model = AMGNNmodel.load_from_checkpoint(model_path)

    return model


def check_files_in_order(arc_dataset: ARCDataset) -> bool:
    """Check that the dataloader files are given in a croissant order (layer order).

    If files are in the croissant order, return True (e.g. file_01, file_05, file_015 -> True).
    Else, return False (e.g. file_10, file_01 , file_11 -> Flase).

    Parameters
    ----------
    arc_dataset: ARCDataset, the dataset to check

    Returns
    -------
    bool, True if the files are in the croissant order, False otherwise

    """
    # List of all file ID
    files_id = list()

    for graph in arc_dataset:
        name = graph.file_name
        file_id = int(name.split("_")[-1])

        files_id.append(file_id)

    # Check if the files_id are in the good order
    in_order = sorted(files_id) == files_id

    return in_order


def configure_dataloader(data_folder: str) -> ARCDataset:
    """Configure a dataloader.

    Parameters
    ----------
    data_folder: str, path to the data folder

    Returns
    -------
    arc_dataset: ARCDataset, pytorch geometric dataset containing the simulation graph
    """
    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops(), T.Distance()])
    arc_dataset = ARCDataset(data_folder, transform=transform)

    # Check the dataset
    assert check_files_in_order(arc_dataset), "arc_dataset do not gave files in a croissant order."

    return arc_dataset


def apply(model: AMGNNmodel, graph: Data):
    y_hat = model.network(graph)
    return y_hat


def process_category_from_one_hot(one_hot: Tensor) -> str:
    """ Transform the process category one hot vector to a comprensible message

    Parameters
    ----------
    one_hot: torch.Tensor, tensor of size 8 containing only zeros with one one.

    Returns
    -------
    category: str, the name corresponding to the tensor category
    """

    one_hot = one_hot.tolist()

    if one_hot == [1, 0, 0, 0, 0, 0, 0, 0]:
        category = "AM_Layer"
    elif one_hot == [0, 1, 0, 0, 0, 0, 0, 0]:
        category = "process"
    elif one_hot == [0, 0, 1, 0, 0, 0, 0, 0]:
        category = "Postcooling"
    elif one_hot == [0, 0, 0, 1, 0, 0, 0, 0]:
        category = "Powderremoval"
    elif one_hot == [0, 0, 0, 0, 1, 0, 0, 0]:
        category = "Unclamping"
    elif one_hot == [0, 0, 0, 0, 0, 1, 0, 0]:
        category = "Cooling - 1"
    elif one_hot == [0, 0, 0, 0, 0, 0, 1, 0]:
        category = "ImmediateXrelease"
    elif one_hot == [0, 0, 0, 0, 0, 0, 0, 1]:
        category = "SupportXremoval"
    else:
        raise KeyError("Category not found")

    return category


def save_graph_to_pt(graph: Data, pt_path: Path):
    """Save a graph as a pt file

    Parameters
    ----------
    graph : Data, torch geometric graph to save
    pt_path : Path, full path with file name
    """

    pt_path.parent.mkdir(parents=True, exist_ok=True)

    graph.to("cpu")
    torch.save(graph.to("cpu"), pt_path)


def simulate_full_print_with_ai(model_path: str, data_path: str, configuration: dict):
    """Simulate a full print just with the results of AI model.

    Will simulate a dataset using AI provided results as past data.
    In the same folder of the model_path, a folder with be created name '{model_path file name} full simulation' and
    simulation files will be saved there.

    A folder is created for each simulation categories (AM_Layer, process, Postcooling...) and .pt and .vtk graph
    simulated are saved there.

    In the '{model_path file name} full simulation' folder, csv files containing the metric results of each simulated
    files and each simulation categories are saved.

    Parameters
    ----------
    model_path: str, path to the .ckpt file
    data_path: str, path ot the folder of dataset (containing the processed folder)
    configuration: configuration dictionary
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Full print simulation with AI using the {device}")

    # Load the trained AI
    model = load_model(model_path)
    model.to(device)

    # Save directory
    save_dir = Path(model_path).parent / (str(Path(model_path).stem) + " full simulation")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Configure the dataloader
    arc_dataset = configure_dataloader(data_path)

    # Set model in evaluation mode
    model.eval()
    with torch.no_grad():
        # Compute the first layer
        past_arc = arc_dataset[0]
        past_arc = past_arc.to(device)
        y_hat = apply(model, past_arc)

        # Get the simulation step category (AM_Layer, powderemoval...)
        actual_step_str = process_category_from_one_hot(past_arc.x[0, 13:21])

        # Store the prediction in the y vector for further use
        past_arc.y = torch.hstack([past_arc.y, y_hat])

        # Save results
        pt_path = save_dir / actual_step_str / f'{past_arc.file_name}.pt'
        save_graph_to_pt(past_arc, pt_path)

        # For all other layer
        for graph in tqdm(arc_dataset[1:], desc="Prediction of the simulation step:"):
            graph = graph.to(device)
            # Get the simulation step category
            actual_step_str = process_category_from_one_hot(graph.x[0, 13:21])

            # Set the previous features to the results of the last prediction
            graph.x = align_x_and_y(past_arc.x.cpu(), y_hat.cpu(), graph.x.cpu(), configuration).to(device)

            # Predict actual layer
            y_hat = apply(model, graph)

            # Actual graph is now the past graph for next iteration
            past_arc = graph

            # Store the prediction in the y vector for further use
            graph.y = torch.hstack([graph.y, y_hat])

            # Save results
            pt_path = save_dir / actual_step_str / f'{graph.file_name}.pt'
            save_graph_to_pt(graph, pt_path)

        # Now analyse the simulation results
        for folder in save_dir.glob("*"):
            if folder.is_dir():
                alignment_error = surface_reconstruction_error(
                    folder=folder,
                    configuration={"scaling_deformation": 0.2})
                alignment_error.to_csv(save_dir / f"alignment_error_{folder.stem}.csv")


if __name__ == "__main__":
    # Load the configuration
    configuration = read_config(Path("configs"))
    model_path = r"..\last.ckpt"
    data_path = r"data\Cubes_light"
    simulate_full_print_with_ai(model_path, data_path, configuration)
