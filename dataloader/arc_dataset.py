import os.path as osp
from pathlib import Path
import torch_geometric as tg
import gzip
import shutil
import logging
import torch
import wandb
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from Simufact_ARC_reader.ARC_CSV import Arc_reader
from dataloader import simulation_files
from utils.merge_ARC import merge_ARC
from tqdm import tqdm
from xml.dom import minidom



class ARCDataset(Dataset):
    """A dataset loading the csv of part, supports and baseplate for each simulation step.
    Each simulation file and step will load 3 csv giving the point cloud of the part."""

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
        self.neighbour_k = neighbour_k
        self.distance_upper_bound = distance_upper_bound

        #: List all simufact folder in raw
        # simufact_folders = self._simufact_folders()
        super().__init__(root, transform, pre_transform)

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
        files = [str(f.name) for f in files]
        return files

    @staticmethod
    def unzip_file(file: Path) -> None:
        """
        Unzip a gz file. This is used as Simufact is saving important xml data in zipped folder.
        The unziped file will be save at the same place of the gz file with the same name without the gz extension.
        :param file: Path to the file to unzip.
        """
        file = Path(file)
        logging.info(f"Dataset preprocessing : Unziping {str(file)}")
        with gzip.open(file, 'rb') as f_in:
            with open(file.parent / file.stem, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    def _simufact_folders(self) -> list[str]:
        """
        List all simulation folders from the raw folder.
        This is useful as the in the raw folder there are non processed Simufact work folder.
        Each folder represent a simulation.
        :return list of all folders name in the raw directory.
        """
        folders_simu = simulation_files.extract_simulation_folder(self.raw_paths)

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
        :return: meta_parameters: dictionary with an entry for every simulation folder containing a dict with all meta parameters.
        """
        meta_parameters = dict()

        # For each simulation
        for sfolder in simufact_folders:
            simulation_parameters = dict()
            # Where the simulation folders are
            spath = Path(self.raw_dir) / Path(sfolder) / Path("Process")

            # Get material
            materiel_path = spath / Path("Material") / Path("material.xml")
            xml_file = minidom.parse(str(materiel_path))
            assert xml_file.lastChild.tagName == 'sfMaterialData'
            simulation_parameters["powder_name"] = xml_file.lastChild.getElementsByTagName('name')[
                0].firstChild.nodeValue

            # Get time
            increment_path = spath / "_Results_" / "Meta" / "Increments.xml"
            # If Increments.xml.gz was not extracted,it should be extracted
            if not increment_path.exists():
                # Unzip the file
                self.unzip_file(increment_path.parent / (increment_path.name + ".gz"))
            # Extract information: time steps
            xml_file = minidom.parse(str(increment_path))
            assert xml_file.firstChild.tagName == 'Increments'
            increments = xml_file.firstChild.getElementsByTagName("Increment")
            increments_dict = dict()
            for increment in increments:
                increment_number = int(increment.getAttribute("Number"))
                increment_time = float(increment.getElementsByTagName("Time")[0].firstChild.nodeValue)
                increments_dict[increment_number] = increment_time
            simulation_parameters["time_step"] = increments_dict

            # Get laser and layer parameters
            build_path = spath / "Stages" / "Build.xml"

            # Extract information: Laser power, speed and layer thickness
            xml_file = minidom.parse(str(build_path))
            assert xml_file.firstChild.tagName == 'stageInfoFile'
            parameter_xml = \
                xml_file.getElementsByTagName("stageInfoFile")[0].getElementsByTagName("stage")[0].getElementsByTagName(
                    "standardParameter")[0]
            parameters_dict = dict()
            parameters_dict["power"] = float(parameter_xml.getElementsByTagName("power")[0].firstChild.nodeValue)
            parameters_dict["speed"] = float(parameter_xml.getElementsByTagName("speed")[0].firstChild.nodeValue)
            parameters_dict["layer_thickness"] = float(
                parameter_xml.getElementsByTagName("layerThickness")[0].firstChild.nodeValue)

            simulation_parameters["printing_parameters"] = parameters_dict

            meta_parameters[sfolder] = simulation_parameters

        return meta_parameters

    def process(self) -> None:
        """
        Process all raw files, work on them to compute the part and supports to generate a graph.
        The graph is then saved as a pt file.
        :return: None
        """
        idx = 0

        #: List all simufact folder in raw
        simufact_folders = self._simufact_folders()
        printing_information = self.get_meta_parameters(simufact_folders)

        tqdm_bar = tqdm( simulation_files.organise_files(self.raw_paths))  # iterate on all files
        for raw_paths in tqdm_bar
            step_arcs = list()
            for raw_path in raw_paths:
                file_name = raw_path.stem
                tqdm_bar.set_description(f"Processing {file_name}")
                # Read data from `raw_path`.
                raw_path = Path(raw_path)

                # Create an Arc_readet object to read the csv and create a graph
                arc = Arc_reader(name=file_name)

                # Read a csv dump
                arc.load_csv(raw_path)

                # Extract the point cloud coordinate
                arc.get_coordinate()

                # Add at each point all extract data
                arc.get_point_cloud_data(display=False)
                step_arcs.append(arc)
            step_name = f"{simulation_files.extract_the_simulation_folder(raw_path)}_{simulation_files.extract_step_folder(raw_path)}"

            # Merge the listed files for this part and simulation step
            arc = merge_ARC(step_arcs)

            # Extract features
            coordinates = torch.tensor(arc.coordinate, dtype=torch.float)
            part_edge_index, length = self.create_edge_list_and_lenght(arc.coordinate)
            part_edge_index = torch.tensor(part_edge_index, dtype=torch.long).t().contiguous()
            length = torch.tensor(length, dtype=torch.float)
            #TODO: How to load the previous step?

            # Extract labels
            # TODO: Add X_DIS, Y_DIS and Z_DIS
            y = torch.tensor(arc.data.TOTDISP, dtype=torch.float)

            # transform into an undirected graph:
            part_edge_index, length = tg.utils.to_undirected(edge_index=part_edge_index, edge_attr=length, reduce='add')

            data = Data(x=coordinates,
                        edge_index=part_edge_index,
                        edge_attr=length,
                        y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, osp.join(self.processed_dir, f'{step_name}.pt'))
                idx += 1

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
    DATA_PATH = "ARC_files"
    arcdataset = ARCDataset(DATA_PATH)

    # Add file name and step to the data?
