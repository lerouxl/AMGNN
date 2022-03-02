from unittest import TestCase
from pathlib import Path
from typing import List
from dataloader.simulation_files import organise_files, extract_simulation_folder, extract_step_folder


def expected_result() -> List[str]:
    expected = [
        [
            Path(r"folderA/Process/_Results_/00000/part.csv"),
            Path(r"folderA/Process/_Results_/00000/supports.csv")
        ],
        [
            Path(r"folderA/Process/_Results_/00001/part.csv"),
            Path(r"folderA/Process/_Results_/00001/supports.csv")
        ],
        [
            Path(r"folderB/Process/_Results_/00002/part.csv",)
        ],
        [
            Path(r"folderC/Process/_Results_/00005/supports.csv")
        ],
        [
            Path(r"folderD/Process/_Results_/00004/part.csv")
        ]
    ]
    return expected


def expected_simulation_names() -> List[str]:
    expected = [
        "folderA",
        "folderB",
        "folderC",
        "folderD"
    ]

    return expected


def expected_steps_names() -> List[str]:
    expected = [
        "00000",
        "00001",
        "00002",
        "00004",
        "00005"
    ]

    return expected


def false_data() -> List[Path]:
    """
    Generate a defined list of files.
    :return: list
    """
    files = [
        r"folderA/Process/_Results_/00000/part.csv",
        r"folderA/Process/_Results_/00000/supports.csv",
        r"folderA/Process/_Results_/00001/part.csv",
        r"folderA/Process/_Results_/00001/supports.csv",
        r"folderB/Process/_Results_/00002/part.csv",
        r"folderB/Process/_Results_/00001/baseplate.csv",
        r"folderC/Process/_Results_/00005/supports.csv",
        r"folderD/Process/_Results_/00004/part.csv",
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

    def test_extract_step_folder(self):
        fdata = false_data()
        results = list()
        for file in fdata:
            results.append(extract_step_folder(file))
        # Remove duplicate
        results = list(set(results))

        self.assertCountEqual(results, expected_steps_names())

