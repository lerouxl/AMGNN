import os.path as osp
from pathlib import Path
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data
#from Simufact_ARC_reader.ARC_CSV import Arc_reader
from tqdm import tqdm
from scipy.spatial import KDTree
import torch_geometric as tg


class ARCDataset(Dataset):
    """A dataset loading the csv of part, supports and baseplate for each simulation step.
    Each simulation file and step will load 3 csv giving the point cloud of the part."""
    def __init__(self, root, mesh_radius, neighbour_k=26, distance_upper_bound = 1.64, transform=None, pre_transform=None):
        """
        When the point cloud is transformed to a graph, the {neighbour_k} points in a sphere are linked by an edge.
        :param root:
        :param mesh_radius:
        :param neighbour_k: Number of points linked to each points during the graph creation. Aka, number of neighbours.
        :param distance_upper_bound
        :param transform:
        :param pre_transform:
        """
        self.mesh_radius = mesh_radius
        self.neighbour_k = neighbour_k
        self.distance_upper_bound = distance_upper_bound

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # List all files that will be processed
        files = list((Path(self.root) / Path("raw")).rglob("*.csv"))
        files = [str(f.name) for f in files]
        return files

    @property
    def processed_file_names(self):
        files = list((Path(self.root) / Path("processed")).rglob("*.pt"))
        files = [str(f.name) for f in files]
        return files

    def process(self):
        idx = 0

        tqdm_bar = tqdm(self.raw_paths)  # iterate on all files
        for raw_path in tqdm_bar:

            file_name = Path(raw_path).stem
            tqdm_bar.set_description(f"Processing {file_name}")
            # Read data from `raw_path`.
            raw_path = Path(raw_path)

            # Create an Arc_readet object to read the csv and create a graph
            arc = None #Arc_reader(name=file_name)

            # Read a csv dump
            arc.load_csv(raw_path)

            # Extract the point cloud coordinate
            arc.get_coordinate()

            # Add at each point all extract data
            arc.get_point_cloud_data(display=False)

            coordinates = torch.tensor(arc.coordinate, dtype=torch.float)
            part_edge_index, lenght = self.create_edge_list_and_lenght(arc.coordinate, radius=self.mesh_radius)
            part_edge_index = torch.tensor(part_edge_index, dtype=torch.long).t().contiguous()
            lenght = torch.tensor(lenght, dtype=torch.float)
            y = torch.tensor(arc.data.TOTDISP, dtype=torch.float)

            # transform into an undirected graph:
            part_edge_index, lenght = tg.utils.to_undirected(edge_index=part_edge_index, edge_attr=lenght, reduce='add')

            data = Data(x=coordinates,
                        edge_index=part_edge_index,
                        edge_attr=lenght,
                        y=y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'{file_name}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data

    ## Compute nearest point
    def create_edge_list_and_lenght(self, coordinates, radius=2e-03):
        """
        take a list of coordinates and compute the neighboors of each coordinates points (edges) and their lenght
        """
        tree = KDTree(coordinates)
        part_edge_index = []
        lenght = []

        # With the query ball point algo
        """for pt_id,pt in enumerate(coordinates):
          neighs = tree.query_ball_point(pt,2e-03)

          for neigh in neighs:
            part_edge_index.append([pt_id, neigh])
            lenght.append(coordinates[pt_id] - coordinates[neigh])

        lenght = norm(lenght, axis=1)"""

        # Query the kd-tree for nearest neighbors.
        for pt_id, pt in enumerate(coordinates):

            distances_neighbors, neighbor_index = tree.query(pt,
                                                             k=self.neighbour_k, eps=0, p=2,
                                                             distance_upper_bound= self.distance_upper_bound, workers=1)

            for neigh in neighbor_index:
                part_edge_index.append([pt_id, neigh])
                lenght.append(coordinates[pt_id] - coordinates[neigh])

        return part_edge_index, lenght

if __name__ == "__main__":
    DATA_PATH = "ARC_files"
    arcdataset = ARCDataset(DATA_PATH, mesh_radius=2e-03)

    # Add file name and step to the data?