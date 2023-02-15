from pathlib import Path
import os
from tqdm import tqdm

arc_tool_exe_path = Path(r"C:\Program Files\simufact\additive\2021\sfTools\sfArcTool\bin\ArcToolCmd.exe")
dataset_path = Path(r"E:\Leopold\Chapter 6 - datasets\Complex_dataset\raw")
arc_files= dataset_path.rglob("*.arc")
arc_files=list(arc_files)

# Move the cmd to the arc toll exe folder
os.chdir(arc_tool_exe_path.parent)
for arc_file in tqdm(arc_files, total=len(arc_files)):
    input_file = arc_file
    output_file = arc_file.with_suffix('.csv')

    os.system(f'ArcToolCmd.exe FileIn="{input_file}" FileOut="{output_file}" Format=4')

print("Finish")
#ArcToolCmd.exe FileIn="C:\my.arc" FileOut="C:\my.csv" Format=4