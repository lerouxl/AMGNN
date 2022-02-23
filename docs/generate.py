from pathlib import Path
import os
folder_to_scan = [".", "dataloader", "model", "ops", "utils"]

for folder in folder_to_scan:
    files = list(Path(folder).rglob("*.py"))
    for file in files:
        os.system(f"conda run -n AMGNN pdoc -o ./docs {file}")
