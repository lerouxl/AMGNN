from unittest import TestCase
from utils.edges_creation import create_edge_list_and_length
from utils.merge_ARC import merge_ARC
from amgnn_unittest.create_test_data import create_two_cubes, create_ARC_object


class TestEdgesCreation(TestCase):
    box1, box2 = create_two_cubes((1, 1, 1), [0, 0, 1.5])
    arc1, arc2 = [create_ARC_object(b, str(i)) for i, b in enumerate([box1, box2])]
    merged_box = merge_ARC([arc1, arc2])

    def test_zeros_edge(self):
        """Test case where they should be no edges"""
        part_edge_index, length = create_edge_list_and_length(neighbour_k=26,
                                                              distance_upper_bound=0.5,
                                                              coordinates=self.arc1.coordinate)
        self.assertEqual(0, len(part_edge_index))

    def test_no_duplicate(self):
        """Check that no duplicated edges are created when not enough close points are available"""
        part_edge_index, length = create_edge_list_and_length(neighbour_k=26,
                                                              distance_upper_bound=1,
                                                              # All vertices are at a distance of 1
                                                              coordinates=self.arc1.coordinate)
        self.assertEqual(24, len(part_edge_index))

    def test_length(self):
        """Check edges size on a simple case"""
        part_edge_index, length = create_edge_list_and_length(neighbour_k=26,
                                                              distance_upper_bound=1,
                                                              coordinates=self.arc1.coordinate)
        for le in length:
            self.assertEqual(1.0, le)

    def test_return_enough_edges(self):
        k = 5
        part_edge_index, length = create_edge_list_and_length(neighbour_k=k,
                                                              distance_upper_bound=10,
                                                              coordinates=self.merged_box.coordinate)

        self.assertEqual(16 * k, len(length))
