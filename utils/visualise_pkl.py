from pathlib import Path

import numpy as np

from utils.save_arc import load_arc
from tqdm import tqdm
from simufact_arc_reader.ARC_CSV import Arc_reader
from utils.visualise import display_x_y_y_hat_data


def load_pkl(file_path: Path) -> Arc_reader:
    """
    Load a pkl arc file.
    """
    arc = load_arc(file_path)
    return arc


if __name__ == "__main__":
    data_path = Path(r"data\test\tmp_arc")
    files = list(data_path.glob("*.pkl"))
    for file in tqdm(files):
        arc = load_pkl(file)
        y = np.stack([arc.data.TEMPTURE, arc.data.XDIS, arc.data.YDIS, arc.data.ZDIS]).transpose()

        display_x_y_y_hat_data(pos=arc.coordinate, x=np.zeros(y.shape), y=y, local_path=file,
                               y_hat=None, display_features=False, config=None)
