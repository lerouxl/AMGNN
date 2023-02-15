"""
This code is scanning all simulation csv files of a folder and extract important features values of AMGNN
For the moment, at the end of this script are displayed:
- the maximal reached temperature in Celsius.
- the maximal deformation in mm (not the norm of the deformation, the max values on all axis).
- all process names founds
- the max number of layers (extract from the process names).


"""
from simufact_arc_reader.ARC_CSV import Arc_reader
from pathlib import Path
from tqdm import tqdm
import numpy as np

dirs = Path(r"E:\Leopold\Chapter 6 - datasets\Complex_dataset\raw").iterdir() # .rglob("*.csv")
dirs = list(dirs)
print(dirs)

max_deformation = 0.0
max_temperature = 0.0
min_temperature = 1000.0
max_coordinate = 0.0
min_coordinate = 10000.0
max_x_time_step = 0.0
min_x_time_step = 10000.0
max_x_time_step_length = 0.0
min_x_time_step_length = 10000.0
all_process_name = {}

# For all simulation folder
for j,dir in enumerate(dirs):
    if dir.is_dir():
        # list all part and supports simulation files
        csv_files = (dir / "_Results_").rglob("*.csv")
        csv_files = list(csv_files)

        # For all simulation file
        pbar = tqdm(enumerate(csv_files), total=len(csv_files))
        for i,file in pbar:
            pbar.set_description(f"Simulation folder {j} / {len(dirs)}")

            # Check that we are working on the dataset
            if not(dir.name in file.name):
                continue

            if ("part" in file.name) or ("supports" in file.name):
                # Load the csv file
                arc = Arc_reader()
                arc.load_csv(file, attribute_to_load= ['Coordinates', 'TEMPTURE', 'XDIS', 'YDIS', 'ZDIS'])

                # Extract the point cloud coordinate
                arc.get_coordinate()
                arc.get_connectivity()

                # Add at each point all extract data
                arc.get_point_cloud_data()

                # Convert data from meter to mm
                arc.coordinate = arc.coordinate * 1000
                arc.data.XDIS, arc.data.YDIS, arc.data.ZDIS = arc.data.XDIS * 1000, arc.data.YDIS * 1000, arc.data.ZDIS * 1000

                # Convert from Kelvin to Celsius
                arc.data.TEMPTURE = arc.data.TEMPTURE - 273.15

                max_deformation = max(max_deformation,
                                      np.abs(arc.data.XDIS.max()),
                                      np.abs(arc.data.YDIS.max()),
                                      np.abs(arc.data.ZDIS.max()),)

                max_temperature = max(max_temperature, arc.data.TEMPTURE.max())
                min_temperature = min(min_temperature, arc.data.TEMPTURE.min())

                max_coordinate = max(max_coordinate, arc.coordinate.max())
                min_coordinate = min(min_coordinate, arc.coordinate.min())


                try:
                    arc.load_meta_parameters(
                        increment_id=i, build_path=None, increments_path=None
                    )

                    try:
                        process_name = str(arc.metaparameters.subProcessName, "utf-8")
                    except:
                        process_name = str(arc.metaparameters.subProcessName)

                    all_process_name[process_name] = " "

                    try:
                        time_steps_s = arc.metaparameters.time_steps_s

                        max_x_time_step = max(max_x_time_step, time_steps_s)
                        min_x_time_step = min(min_x_time_step, time_steps_s)
                    except:
                        pass

                    try:
                        time_steps_length_s = arc.metaparameters.time_steps_length_s

                        max_x_time_step_length = max(max_x_time_step_length, time_steps_length_s)
                        min_x_time_step_length = min(min_x_time_step_length, time_steps_length_s)
                    except:
                        pass

                except:
                    pass

print(f"The maximal temperature found is {max_temperature} C")
print(f"The minimal temperature found is {min_temperature} C")
print(f"The maximal displacement found is {max_deformation}")
print(f"The maximal coordinate found is {max_coordinate} mm")
print(f"The minimal coordinate found is {min_coordinate} mm")
print(f"The maximal time step found is {max_x_time_step}")
print(f"The minimal time step found is {min_x_time_step}")
print(f"The maximal time step found length is {max_x_time_step_length}")
print(f"The minimal time step found length is {min_x_time_step_length}")
print(f"Here are all found process name: {all_process_name.keys()}")