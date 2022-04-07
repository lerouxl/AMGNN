"""Test features.py"""

import unittest
from utils.merge_ARC import merge_arc
from utils import features
from amgnn_unittest.create_test_data import create_two_cubes, create_ARC_object

class Test_features(unittest.TestCase):
    box1, box2 = create_two_cubes((1, 1, 1), [0, 0, 1.5])
    arc1 = create_ARC_object(box1, "test_part")
    arc2 = create_ARC_object(box2, "test_supports")
    merged_box = merge_arc([arc1, arc2])

    def test_number_of_neighbours(self):
        """
        This function will take two arc reader objects, pass them in the features.arc_features_extraction and check the
        numbers of neighbours on each element of the tensor.
        :return:
        """
        coordinates, part_edge_index, length, x, y = \
            features.arc_features_extraction(arc=self.merged_box, neighbour_k=26, distance_upper_bound=1)

        self.assertEqual(part_edge_index.shape[0], 26)

