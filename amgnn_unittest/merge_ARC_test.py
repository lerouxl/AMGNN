"""Test merge_ARC.py"""

import unittest
from utils.merge_ARC import merge_arc
from Simufact_ARC_reader.ARC_CSV import Arc_reader
import numpy as np
from amgnn_unittest.create_test_data import create_two_cubes, create_ARC_object


class Test_merge_ARC(unittest.TestCase):
    box1, box2 = create_two_cubes((1, 1, 1), [0, 0, 1.5])
    arc1, arc2 = [create_ARC_object(b, str(i)) for i, b in enumerate([box1, box2])]
    merged_box = merge_arc([arc1, arc2])

    def test_number_points(self):
        """
        Create 2 cubes, load them in 2 dummy Arc_reader object and try to merge them.
        Check the number of point

        :return:
        """
        self.assertEqual(16, self.merged_box.coordinate.shape[0])

    def test_data(self):
        """
        Check that the merge_ARC did not shuffle the data.
        :return:
        """
        # For the 2 original arc object
        for arc_origin in [self.arc1, self.arc2]:
            # At every points
            for i, _ in enumerate(arc_origin.coordinate):
                coor = arc_origin.coordinate[i]
                # Found the equivalent point on the merged arc
                for j, mcoor in enumerate(self.merged_box.coordinate):
                    if (mcoor == coor).all():
                        break

                self.assertTrue((arc_origin.coordinate[i] == self.merged_box.coordinate[j]).all(),
                                 f"The coordiate of ARC origine and merged ARC are not the same")
                self.assertTrue((arc_origin.data.YDIS[i] == self.merged_box.data.YDIS[j]).all(), "YDIS is not ok")
                self.assertTrue((arc_origin.data.ZDIS[i] == self.merged_box.data.ZDIS[j]).all(), "ZDIS is not ok")
                self.assertTrue((arc_origin.data.XDIS[i] == self.merged_box.data.XDIS[j]).all(), "XDIS is not ok")
                self.assertTrue((arc_origin.data.TEMPTURE[i] == self.merged_box.data.TEMPTURE[j]).all(), "TEMPTURE is not ok")

# if __name__ == '__main__':
#    unittest.main()
