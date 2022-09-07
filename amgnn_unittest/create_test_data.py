import trimesh
from typing import Tuple, List
from simufact_arc_reader.ARC_CSV import Arc_reader
import numpy as np

def create_two_cubes(size: Tuple = (1, 1, 1), space: List = [0, 0, 1.5]):
    """
    Create 2 box of size size and move on using the vector space.
    :param size: Size of the cubes
    :param space: Vector to move the second box
    :return:
    """
    box1 = trimesh.primitives.Box(extents=size)
    box2 = trimesh.primitives.Box(extents=size)
    box2 = trimesh.base.Trimesh(box2.vertices + space, box2.faces)

    return box1, box2

def create_ARC_object(mesh, name:str ="arc_file") -> Arc_reader:
    """
    Create an ARC object from a trimesh mesh.
    The ARC object will be filled with the mesh vertices coordinates.
    The data are filled with random informations.
    :param mesh: Trimesh object
    :param name: Name of the ARC object
    :return: Arc reader object
    """
    arc_obj = Arc_reader(name)
    # Give the fake object the type extrcated from it's name
    # It can be supports, part or baseplate
    arc_obj.arc_type = arc_obj._get_arc_type(name)
    arc_obj.coordinate = mesh.vertices
    numbers_of_vertices = arc_obj.coordinate.shape[0]
    # Generate dummy data
    names = ["XDIS", "YDIS", "ZDIS", "TEMPTURE"]
    for name in names:
        # Create fake data in function of their name
        if name == "TEMPTURE":
            data = np.random.uniform(low=700, high=1500, size=numbers_of_vertices)
        elif name in ["XDIS", "YDIS", "ZDIS"]:
            data = np.random.uniform(low=-1, high=1, size=numbers_of_vertices)
        else:
            data = np.random.uniform(low=0, high=10, size=numbers_of_vertices)
        setattr(arc_obj.data, name, data)

    # Generate meta data
    arc_obj.metaparameters.layerThickness_m = np.random.randint(0,10) * 1e-5
    arc_obj.metaparameters.power_W = np.random.randint(1,4) * 100.0
    arc_obj.metaparameters.process_step = np.random.randint(0,100)
    arc_obj.metaparameters.speed_m_s = np.random.randint(5,15) * 0.1
    arc_obj.metaparameters.time_steps_length_s = np.random.random(1)[0]
    arc_obj.metaparameters.time_steps_s = np.random.random(1)[0] * 1000

    return arc_obj

if __name__ == "__main__":
    boxs = create_two_cubes()
    create_ARC_object(boxs[0])