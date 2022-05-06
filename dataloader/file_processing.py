import os.path as osp
import torch_geometric as tg
import logging
import torch
from torch_geometric.data import Data
from utils.save_arc import load_arc
from utils import features
import logging


def processing_file(raw_path: str, processed_dir: str, conf_wandb: dict) -> None:
    """
    Load a preprocessed file (plk) and create a graph from it.
    The graph is then save under the same name in the processed_dir directory.
    :param raw_path: plk file path
    :param processed_dir: Where to save the graph
    :return: None
    """
    log = logging.getLogger(__name__)
    log.info(f"Start processing_file on file {str(raw_path)}")
    arc = load_arc(raw_path)
    log.debug("File loaded")

    # Stop if this is the initialisation file
    if arc.is_first_step:
        log.debug("The ARC file was a first simulation step, skip.")
        return None

    # Extract features
    coordinates, part_edge_index, length, X, Y = features.arc_features_extraction(arc=arc, \
                                                                                  past_arc=arc.previous_arc, \
                                                                                  config=conf_wandb)
    log.debug(f"Features extracted from {str(raw_path.stem)}, the previous simulation step arc was {arc.previous_file_name}")
    log.debug(f"{coordinates.shape[0]} points were extracted. With {part_edge_index.shape[1]} edges and {length.shape[0]} edge attributes")
    print(f"{coordinates.shape[0]} points were extracted. With {part_edge_index.shape[1]} edges and {length.shape[0]} edge attributes")

    # transform into an undirected graph:
    part_edge_index, length = tg.utils.to_undirected(edge_index=part_edge_index, edge_attr=length)
    log.debug("Undirected graph created")

    data = Data(x=X.rename(None),
                edge_index=part_edge_index.rename(None),
                edge_attr=length.rename(None),
                y=Y.rename(None),
                pos=coordinates.rename(None))
    log.debug("Data object created")

    torch.save(data, osp.join(processed_dir, f'{raw_path.stem}.pt'))
    log.debug("Data object saved")
    return None
