from pathlib import Path
import gzip
import shutil
import logging
from xml.dom import minidom


def unzip_file(file: Path) -> None:
    """Unzip a gz file.

    This is used as Simufact is saving important xml data in zipped folder.
    The unziped file will be save at the same place of the gz file with the same name without the gz extension.

    Parameters
    ----------
    file: Path
        Path to the file to unzip.
    Returns
    -------
    None :
        The gz file is saved next to its compressed file.
    """
    file = Path(file)
    logging.info(f"Dataset preprocessing : Unziping {str(file)}")
    with gzip.open(file, 'rb') as f_in:
        with open(file.parent / file.stem, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def get_simulation_parameters(spath: Path) -> dict:
    """ Extract the simulation parameters for a simulation.

    Using the simulation root folder, this function will extract all material, time, laser, layer and speed parameters.
    The printing parameters are saved in a dictionary and returned.

    Parameters
    ----------
    spath: Path
        Simulation root folder.

    Returns
    -------
    dict:
        A dictionary with the printing parameters as key and their values.

    """
    simulation_parameters = dict()
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
        unzip_file(increment_path.parent / (increment_path.name + ".gz"))

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

    return simulation_parameters
