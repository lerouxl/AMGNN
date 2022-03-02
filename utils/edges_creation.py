import numpy as np
from scipy.spatial import KDTree


def create_edge_list_and_length(neighbour_k: int, distance_upper_bound: float, coordinates: np.array):
    """
    Take a list of coordinates and compute the neighbours of each coordinates points (edges) and their length
    using the KD tree query.
    The graph created with this do not have a self loop.
    :param neighbour_k: Number of neighbors/edges per vertices
    :param distance_upper_bound: Max distance to look up
    :param coordinates: coordinates of every points
    """
    tree = KDTree(coordinates)
    part_edge_index = []
    length = []
    epsilon = 1e-10
    neighbour_k += 1 # By default, query list also the self vertex,
    # it will be removed after so we have to a 1 to have the correct number of edges

    # Query the kd-tree for nearest neighbors.
    for pt_id, pt in enumerate(coordinates):

        distances_neighbors, neighbor_index = tree.query(pt,
                                                         k=neighbour_k, eps=0, p=2,
                                                         distance_upper_bound=distance_upper_bound + epsilon, workers=1)

        for dist, neigh in zip(distances_neighbors, neighbor_index):
            if (dist == float("inf")) or (dist == 0):
                # Remove points further than the distance_upper_bound (inf) and the point itself.
                continue
            # If the point is already listed as a neighbor, we do not list it again.
            if pt_id in part_edge_index:
                continue

            part_edge_index.append([pt_id, neigh])
            length.append(dist)

    return part_edge_index, length
