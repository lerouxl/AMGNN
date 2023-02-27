import os.path as osp
from pathlib import Path
import torch
from torch_geometric.data import Dataset, Data
from dataloader import simulation_files
from dataloader.file_preprocessing import preprocess_folder
from dataloader.printing_parameters import get_simulation_parameters
from dataloader.file_processing import processing_file
from itertools import repeat
import wandb
from tqdm import tqdm
from multiprocessing import Pool

class ARCDataset(Dataset):
    """A dataset loading the csv of part and supports for each simulation step.

    This :obj:Dataset object is containing the graph of a simulation step. In each step, the printed voxels are
    represented by a datapoint linked together using a graph structure.
    Each nodes of the graph contain the following information.
    TODO: Update ARCDataset X tensors description, we now have 22 values.
    Data:
        - X: 22 features for each nodes used for predict Y
            - Laser speed (m/s scaled with `scaling_speed`)
            - Laser power (W scaled with `scaling_power`)
            - Layer thickness (um/100)
            - Time step length (s  scaled by `100 000`)
            - Time step (s scaled with `scaling_time`)
            - Type: One hot vector for ["baseplate", "part", "supports"]
            - Past temperature (in Celsius scaled with `scaling_temperature`)
            - Past displacement X (in `mm` scaled with `scaling_deformation`)
            - Past displacement Y (in `mm` scaled with `scaling_deformation`)
            - Past displacement Z (in `mm` scaled with `scaling_deformation`)
            - Process features (number of printed voxel layer scaled with `max_number_of_layer`)
            - Process category: One hot vector for [AM_Layer, process, Postcooling, Powderromval, Unclamping, Cooling-1]
            - Coordinate X (in `mm` scaled with scaling_size)
            - Coordinate Y (in `mm` scaled with scaling_size)
            - Coordinate Z (in `mm` scaled with scaling_size)
        - Y (target):
            - TEMPTURE (in Celsius scaled with `scaling_temperature`)
            - XDIS (in `mm` scaled with `scaling_deformation`)
            - YDIS (in `mm` scaled with `scaling_deformation`)
            - ZDIS (in `mm` scaled with `scaling_deformation`)
        - pos: Tensor of shape [n, 3] containing the non deformed position of the voxels (in `mm` scaled with scaling_size).

    Notes: All values are scaled between 0 and 1 using the configuration scaling variables.
    """

    def __init__(self, root: Path, transform=None, pre_transform=None):
        """
         Will load simulation file and transform them to graph.

        Parameters
        ----------
        root : Path
            Path containing the raw and processed data folders.
        transform : :obj:, optional
            A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform: :obj:, optional
            A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        """

        if type(root) is str:
            root = Path(root)

        self.tmp_arc_folder = root / "tmp_arc"

        #: List all simufact folder in raw
        # simufact_folders = self._simufact_folders()
        super().__init__(str(root), transform, pre_transform)

    @property
    def raw_file_names(self) -> list[Path]:
        """ List raw files available.

        List all ARC csv files in the config root raw folder.
        All csv files must be an extract of the ARC file generated using Simufact tool.

        Returns
        -------
        list of :obj:`Path`
            List of obj:`Path` to the .csv in the raw directory *raw_dir*.
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
    def processed_file_names(self) -> list[str]:
        """ Return name of already processed files.

        List all processed files by the dataloader saved as pt files in the processed directory.

        Returns
        -------
        list of :obj:`str`
            List of all files name saved as pt files (except for pre_filter and pre_transform)
        """
        files = list((Path(self.root) / Path("processed")).rglob("*.pt"))
        # Pre_filter and pre_transform are not data for the neural network, so I removed them form the list of inputs
        # files
        files = [str(f.name) for f in files if str(f.name) not in ['pre_filter.pt', 'pre_transform.pt']]
        return files

    def _simufact_folders(self) -> list[str]:
        """ List all simulation folders from the raw folder.

        This is useful as the in the raw folder there are non processed Simufact work folder.
        Each folder represent a simulation.

        Returns
        -------
        list of :obj:`str`
            The list of all folders name in the raw directory.
        """
        self.all_arc_files = list(Path(self.raw_dir).rglob("*_FV_*.csv"))
        folders_simu = simulation_files.extract_simulation_folder(self.all_arc_files)

        if "Template" in folders_simu:
            folders_simu.remove("Template")

        return folders_simu

    def get_meta_parameters(self, simufact_folders: list[Path]) -> dict:
        """ Get the simulation parameters used for the simulation.

        This function will go in all of the Simufact simulation folder and get information such as:
        Information:
            - material name : str, name of the material used by the simulation
            - Type (part, supports): one hot vector
            - Laser power (W)
            - Laser speed
            - Layer thickness (mm)
            - Layer processing time: float (s), time since the launch of the print (one for each step)
            - Past node temperature or Build chamber temperature for new node
            - Past node displacement or 0 for new nodes
            - Coordinates
            - In contact with the baseplate.

        Parameters
        ----------
        simufact_folders: list of Path
            The list of all simulation folder in raw folder.

        Returns
        -------
        dict
            meta_parameters: dictionary with an entry for every simulation folder containing a dict with all meta.

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
        """ Process raw files to pt files.

        Process all raw files, work on them to compute the part and supports to generate a graph.
        The graph is then saved as a pt file.

        Returns
        -------
        None

        """
        #: List all simufact folder in raw
        simufact_folders = self._simufact_folders()
        # printing_information = self.get_meta_parameters(simufact_folders)

        # Merge same simulation/step ARC csv and save the results in a temporary folder.
        # The ARC_reader object is saved with pickle

        # Pre processing
        # TODO: Do a proper test if the preporcessing is done
        # preprocessing_done = len(list(self.tmp_arc_folder.glob("*.npz"))) > 0
        preprocessing_done = wandb.config.preprocessing_done  # Controlled now with the config_data.yml

        if not preprocessing_done:
            # Is the multiprocessing boolean is set to True, then use pooling_process_th pools to preprocess the data
            if wandb.config.do_multi_processing:
                with Pool(wandb.config.pooling_process) as pool:
                    pool.starmap(preprocess_folder, zip(simufact_folders,
                                                        repeat(self.all_arc_files),
                                                        repeat(self.tmp_arc_folder)))
            else:
                for simu in tqdm(simufact_folders):
                    preprocess_folder(simu, self.all_arc_files, self.tmp_arc_folder)

        # Processing
        tmp_files = list(self.tmp_arc_folder.glob("*.npz"))
        #with Pool(wandb.config.pooling_process) as pool:
        #    # If distance_upper_bound is too low, error can append in the processing
        #    pool.starmap(processing_file, zip(tmp_files,
        #                                      repeat(self.processed_dir),repeat(dict(wandb.config))))
        # Without multi processing (can be entered with the debugger
        #log = logging.getLogger(__name__)
        #log.info(f"Start processing tmp files")
        for data in zip(tmp_files,repeat(self.processed_dir),repeat(dict(wandb.config))):
            #print()
            #try:
            processing_file(*data)
        #    except Exception as e:
        #        s = str(e)
        #        log.critical(f"FAIL: {s} ")
        #        log.critical(f"FAIL: {str(data[0])}")



    def len(self) -> int:
        """ Number of processed files.

        Will list all *.pt* files saved in the processed directory, and count them.

        Returns
        -------
        int
            The number of processed files
        """
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        """ Get a graph.

        Get the `idx`.pt file and load it as a torch geometric graph.

        Parameters
        ----------
        idx: int
            ID of the file to load.

        Returns
        -------

        Data
            The graph of the loaded pt file.
        """
        file = self.processed_file_names[idx]
        data = torch.load(osp.join(self.processed_dir, file))
        # Add the name of the file to the graph
        # This is used in the test function where results are saved in pt file
        data.file_name = Path(file).stem
        return data


if __name__ == "__main__":
    from utils.config import read_config

    # Initialise wandb
    configuration = read_config(Path("configs"))
    wandb.init(mode="offline", config=configuration)
    DATA_PATH = Path("data/light")
    arcdataset = ARCDataset(DATA_PATH)

    # Add file name and step to the data?
