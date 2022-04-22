from unittest import TestCase
import torch
from utils import align_features
import numpy as np


class Test_align_features(TestCase):
    # Genrate dumy data:

    # The small arc file (previous simulation step)
    number_of_points = 10
    small_range = torch.arange(start=1, end=number_of_points + 1)
    np.random.shuffle(small_range.numpy())
    past_features = torch.unsqueeze(small_range, 1).expand(10, 7)
    past_features = past_features - torch.tensor([[0, 0, 0, 1, 1, 1, 1]]).expand(10, 7)

    # The big arc file (actual simulation step)
    big_range = torch.arange(start=1, end=2 * number_of_points + 1)
    actual_features = torch.unsqueeze(big_range, 1).expand(20, 14)

    fake_config = {
        "distance_upper_bound": 0.5,
        "default_powder_temperature": 100,
        "scaling_temperature": 1
    }

    # scale deformation by dividing it by 10
    past_scaling_tensor = torch.tensor([1,1,1,1,0.1,0.1,0.1]).expand(10,7)
    actual_scaling_tensor = torch.tensor([*[1 for _ in range(14-3)], 0.1,0.1,0.1])

    actual_features = actual_features * actual_scaling_tensor
    past_features = past_features * past_scaling_tensor

    actual_coor_no_def, X, Y = align_features.align_features(actual_features=actual_features,
                                                             past_features=past_features,
                                                             config=fake_config)

    def test_new_points(self):
        """Check that the new point are initialised with a deformation of 0 and a temperature of default powder tempt
        / scale """
        expected_default_val = torch.tensor([float(self.fake_config["default_powder_temperature"]) /
                                             float(self.fake_config["scaling_temperature"]), 0, 0, 0])
        for line in range(self.number_of_points + 1, 2 * self.number_of_points ):
            self.assertTrue(torch.eq(self.X[line][-4:], expected_default_val).all(), "New points are badly initialised")

    def test_old_point(self):
        """Check that the point have their past value"""

        for line in range(0 , self.number_of_points):
            deformation = line * 0.1
            temperature = line
            expected_old_val = torch.tensor([temperature, deformation, deformation, deformation])
            result = self.X[line][-4:].rename(None)

            tensor_equal = torch.isclose(result, expected_old_val).all()
            self.assertTrue(tensor_equal,"Old points did not take the previous value")