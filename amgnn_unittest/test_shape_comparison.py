import unittest
from utils.shape_comparison import graph_error
import torch
import torch_geometric as tg
from torch_geometric.data import Data
import pyvista as pv


class Test_shape_comparison(unittest.TestCase):

    def test_no_difference(self):
        """
        Gave twice the same deformation and expect to have an error equal to zeros.
        """
        cube = pv.Cube((0, 0, 0), 10, 10, 10)
        cube = pv.voxelize(cube, 4)

        graph = Data(pos=torch.Tensor(cube.points),  # Create a graph with point at every cube vertex
                     y=torch.ones((len(cube.points), 8)),  # Gave a vector 1,1,1 as label and results
                     )

        results = graph_error(graph, scaling_deformation=1, m_to_mm=False)

        max_error = results["Distances from simulation (mm)"].max()
        min_error = results["Distances from simulation (mm)"].min()
        mean_error = results["Distances from simulation (mm)"].mean()

        self.assertAlmostEqual(max_error, 0, msg="The errors should be equal to zeros")
        self.assertAlmostEqual(min_error, 0, msg="The errors should be equal to zeros")
        self.assertAlmostEqual(mean_error, 0, msg="The errors should be equal to zeros")

    def test_one_difference(self):
        """
        Except to have an error equal to one.
        In this case, the hypothesis is made that AMGNN results is 10% bigger than a cubes.
        """
        cube = pv.Cube((0, 0, 0), 5, 5, 5)

        # Generate a deformation vector by scaling the points and using them to calculate a deformation vectors
        cube2 = pv.Cube((0, 0, 0), 6, 6, 6)

        deformation = cube.points - cube2.points
        deformation = torch.Tensor(deformation)
        tensor_deformation = torch.hstack(
            (torch.zeros((len(cube.points), 5)),  # Zero temp, zero deformation and zeros temperature
             deformation)
            )

        graph = Data(pos=torch.Tensor(cube.points),  # Create a graph with point at every cube vertex
                     y=tensor_deformation,  # Gave a vector 1,1,1 as label and results
                     )

        results = graph_error(graph, scaling_deformation=1, m_to_mm=False)

        max_error = results["Distances from simulation (mm)"].max()
        min_error = results["Distances from simulation (mm)"].min()
        mean_error = results["Distances from simulation (mm)"].mean()

        """
        p = pv.Plotter()
        p.add_mesh(cube, show_edges=True, opacity=0.5)
        p.add_mesh(cube2, color="lightblue", show_edges=True, opacity=0.5)
        p.add_mesh(results)
        p.show()
        results.save("test.vtk")"""

        self.assertAlmostEqual(max_error, -1, msg="The errors should be equal to one")
        self.assertAlmostEqual(min_error, -1, msg="The errors should be equal to one")
        self.assertAlmostEqual(mean_error, -1, msg="The errors should be equal to one")
