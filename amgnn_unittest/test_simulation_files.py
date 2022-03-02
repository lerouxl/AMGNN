from unittest import TestCase
from pathlib import Path
from dataloader.simulation_files import organise_files, extract_simulation_folder


def expected_result():
    expected = [
        [
            r"folderA/00000/part.csv",
            r"folderA/00000/supports.csv"
        ],
        [
            r"folderA/00001/part.csv",
            r"folderA/00001/supports.csv"
        ],
        [
            r"folderB/00000/part.csv",
        ],
        [
            r"folderC/00000/supports.csv"
        ],
        [
            r"folderD/00000/part.csv"
        ]
    ]
    return expected


def expected_simulation_names():
    expected = [
        "folderA",
        "folderB",
        "folderC",
        "folderD"
    ]

    return expected


def false_data():
    """
    Generate a defined list of files.
    :return: list
    """
    files = [
        r"folderA/Process/_Results_/00000/part.csv",
        r"folderA/Process/_Results_/00000/supports.csv",
        r"folderA/Process/_Results_/00001/part.csv",
        r"folderA/Process/_Results_/00001/supports.csv",
        r"folderB/Process/_Results_/00000/part.csv",
        r"folderB/Process/_Results_/00000/baseplate.csv",
        r"folderC/Process/_Results_/00000/supports.csv",
        r"folderD/Process/_Results_/00000/part.csv",
    ]
    files = [Path(f) for f in files]
    return files


class Test(TestCase):

    def test_organise_files(self):
        result = organise_files(false_data())
        self.assertCountEqual(result, expected_result(),
                              "simulation_files.organise_files didn't not filter the input list correctly.")

    def test_extract_simulation_folder(self):
        result = extract_simulation_folder(false_data())
        self.assertCountEqual(result, expected_simulation_names())
