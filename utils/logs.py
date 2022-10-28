import logging
from torch_geometric.data import Data
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import wandb
from pathlib import Path


class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)[:, :3] * 255


def init_logger(file_name: str) -> None:
    """
    Configure the logging library.
    logger can be now crated using `log = logging.getLogger(__name__)`
    """
    logging.basicConfig(
        filename=file_name,
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def log_point_cloud_to_wandb(name: str, points: np.array, value: np.array, epoch_number: int, max_value: int = 1):
    """Log a point cloud to wandb

    Parameters
    ----------
    name: str
        Name of the points cloud to save on wandb
    points: numpy.array
        X,Y,Z coordinates of the points cloud. Of shape [n,3]
    value: numpy.array
        Value per nodes to be transformed into an RGB value, shape [n,1]
    epoch_number: int
        Actual epoch
    max_value:
        Max value to use for scaling for scaling

    Returns
    -------

    """
    COL = MplColorHelper('jet', 0, max_value)

    wandb.log(
        {"epoch": epoch_number,
         name: wandb.Object3D(
             {
                 "type": "lidar/beta",
                 "points": np.hstack([points, COL.get_rgb(value)])

             }
         )
         })
