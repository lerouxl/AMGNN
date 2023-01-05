"""
This code is used to process a dataset and remove the categorical features describing the simulation step such as:
[AM_Layer, process, Postcooling, Powderromval, Unclamping, Cooling-1, ImmediateXrelease]
by removing all simulation step other than AM_Layer.
"""
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a dataset to only keep AM_Layer steps."
    )
    parser.add_argument("-i", "--input", type=str,
                        help="Path to the inputs folder where there is the processed pt files")
    parser.add_argument("-o", "--output", type=str,
                        help="Where will be saved the new pt files. If do not exist will be created")

    args = parser.parse_args()

    input_folder = Path(args["input"])
    output_folder = Path(args["output"])
    # Create folder if do not exist
    output_folder.mkdir(parents=True, exist_ok=True)

    input_files = input_folder.glob("*step*.pt")

    for input_file in tqdm(input_files):
        file = torch.load(input_file)

        # Test if this file is from an AM_layer step:
        is_am_layer = bool((file.x[0, 13:-3] == torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])).all())

        if is_am_layer:
            file.x = torch.hstack((file.x[:, :13], file.x[:, -3:]))
            torch.save(file, output_folder / input_file.name)
