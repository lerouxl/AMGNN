from unittest import TestCase
from pathlib import Path
from dataloader.simulation_files import organise_files


class Test(TestCase):
    def false_data(self):
        """
        Generate a defined list of files.
        :return: list
        """
        files = [
            r"folderA/00000/part.csv",
            r"folderA/00000/supports.csv",
            r"folderA/00001/part.csv",
            r"folderA/00001/supports.csv",
            r"folderB/00000/part.csv",
            r"folderB/00000/baseplate.csv",
            r"folderC/00000/supports.csv",
            r"folderD/00000/part.csv",
        ]
        files = [Path(f) for f in files]
        return files

    def expected_result(self):
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

    def test_organise_files(self):
        result = organise_files(self.false_data())
        self.assertCountEqual(result, self.expected_result(), "simulation_files.organise_files didn't not filter the input list correctly.")

