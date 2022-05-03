import os.path as osp
from pathlib import Path
import torch
from torch_geometric.data import Dataset
from dataloader import simulation_files
from dataloader.file_preprocessing import preprocess_folder
from dataloader.printing_parameters import get_simulation_parameters
from dataloader.file_processing import processing_file
from itertools import repeat
from multiprocessing import Pool


class ARCDataset(Dataset):
    """A dataset loading the csv of part, supports and baseplate for each simulation step.
    Each simulation file and step will load 3 csv giving the point cloud of the part.
    The given graph are providing the following data:
        - X:  [n, 11] : laser speed,  laser power, layer thickness, time step duration, x_type,
                        x_type, x_type, scalled past temperature, x past displacement,
                        y past displacement, z past displacement.
        - Y: [n, 4] : scaled actual temperature, actual x displacement , actual y displacement, actual z displacement
        - pos: [n, 3] : The non deformed position of the voxels
        - edge_attr: [n] : The distance between the two linked nodes (non deformed).
    """

    def __init__(self, root: Path, neighbour_k: int = 26, distance_upper_bound: float = 1.64, transform=None,
                 pre_transform=None):
        """
        When the point cloud is transformed to a graph, the `neighbour_k` points in a sphere are linked by an edge.
        :param root: Path containing the raw and processed data folders.
        :param neighbour_k: Number of points linked to each points during the graph creation. AKA, number of neighbours.
        :param distance_upper_bound: float. The found neighbours are closer than this distance.
        :param transform:
        :param pre_transform:
        """
        if type(root) is str:
            root = Path(root)

        self.neighbour_k = neighbour_k
        self.distance_upper_bound = distance_upper_bound
        self.tmp_arc_folder = root / "tmp_arc"

        #: List all simufact folder in raw
        # simufact_folders = self._simufact_folders()
        super().__init__(str(root), transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        List all ARC csv files in the config root raw folder.
        All csv files must be an extract of the ARC file generated using Simufact tool.
        :return: list of all csv path files from the raw_dir.
        """
        # List all files that will be processed
        files = list()
        # List all files that are an used components
        for components in wandb.config.used_components:
            files.extend(list(Path(self.root / Path("raw")).rglob(f"*{components}*.csv")))
        # files = [str(f.name) for f in files]
        files = [f.relative_to(self.raw_dir) for f in files]
        return files

    @property
    def processed_file_names(self):
        """
        List all processed files by the dataloader. Those file where saved as pt files.
        :return: list of files name
        """
        files = list((Path(self.root) / Path("processed")).rglob("*.pt"))
        # Pre_filter and pre_transform are not data for the neural network, so I removed them form the list of inputs
        # files
        files = [str(f.name) for f in files if str(f.name) not in ['pre_filter.pt', 'pre_transform.pt']]
        return files

    def _simufact_folders(self) -> list[str]:
        """
        List all simulation folders from the raw folder.
        This is useful as the in the raw folder there are non processed Simufact work folder.
        Each folder represent a simulation.
        :return list of all folders name in the raw directory.
        """
        self.all_arc_files = list(Path(self.raw_dir).rglob("Process_FV_*.csv"))
        folders_simu = simulation_files.extract_simulation_folder(self.all_arc_files)

        return folders_simu

    def get_meta_parameters(self, simufact_folders: list[Path]) -> dict:
        """
        Will go in all the simufact simluation folder and get information such as:
            - material name : str, name of the material used by the simulation
            - Type (part, supports): one hot vector
            - Laser power (W)
            - Laser speed
            - Layer thickness (mm)
            - Layer processing time: float (s), time since the launch of the print (one for each step)
            - Past node temperature or Build chamber temperature for new node
            - Past node displacement or 0 for new nodes
            - Coordinates
            - In contact with the baseplate

        :param simufact_folders: list of all simulation folder in raw folder.
        :return: meta_parameters: dictionary with an entry for every simulation folder containing a dict with all meta
        parameters.
        """
        meta_parameters = dict()

        # For each simulation
        for sfolder in simufact_folders:
            # Where the simulation folders are
            process_path = Path(self.raw_dir) / Path(sfolder) / Path("Process")

            simulation_parameters = get_simulation_parameters(process_path)

            meta_parameters[sfolder] = simulation_parameters

        return meta_parameters

    def process(self) -> None:
        """
        Process all raw files, work on them to compute the part and supports to generate a graph.
        The graph is then saved as a pt file.
        :return: None
        """
        #: List all simufact folder in raw
        simufact_folders = self._simufact_folders()
        # printing_information = self.get_meta_parameters(simufact_folders)

        # Merge same simulation/step ARC csv and save the results in a temporary folder.
        # The ARC_reader object is saved with pickle

        # Pre processing
        preprocessing_done = len(list(self.tmp_arc_folder.glob("*.pkl"))) > 0
        if not preprocessing_done:
            with Pool(wandb.config.pooling_process) as pool:
                pool.starmap(preprocess_folder, zip(simufact_folders, repeat(self.all_arc_files),
                                                    repeat(self.tmp_arc_folder)))
            # for simu in tqdm(simufact_folders):
            #    preprocess_folder(simu, self.all_arc_files, self.tmp_arc_folder)

        # Processing
        tmp_files = list(self.tmp_arc_folder.glob("*.pkl"))
        with Pool(wandb.config.pooling_process) as pool:
            pool.starmap(processing_file, zip(tmp_files,
                                              repeat(self.processed_dir),
                                              repeat(dict(wandb.config))))

    def len(self) -> int:
        """
        Return the number of processed files (pt).
        :return: int, number of processed files
        """
        return len(self.processed_file_names)

    def get(self, idx: int) -> torch.Tensor:
        """
        Get the `idx`.pt file and load it as a torch variable.
        :param idx: int, id of the file
        :return: The loaded pt file.
        """
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data


if __name__ == "__main__":
    import wandb
    from utils.config import read_config

    # Initialise wandb
    configuration = read_config(Path("configs"))
    wandb.init(mode="offline", config=configuration)
    DATA_PATH = Path("data/light")
    arcdataset = ARCDataset(DATA_PATH)

    # Add file name and step to the data?
