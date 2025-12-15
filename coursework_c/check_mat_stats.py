import zipfile
from pathlib import Path
import scipy.io as spio 
import numpy as np


def import_dataset(filepath):
    mat = spio.loadmat(str(filepath), squeeze_me=True)
    return mat


def print_data(mat_data):
    for idx, _dict in enumerate(mat_data):
        print(f"D{idx+1} data: ")
        for _class in range(5):
            count = np.sum(_dict["Class"] == _class + 1)
            print(f"Class {_class+1}: {count} spikes")


if __name__ == "__main__":
    
    cwd = Path.cwd()
    output_dir = Path(cwd, "coursework_c", "outputs")

    files_to_check = [fp for fp in output_dir.iterdir() if fp.suffix == ".zip"]
    temp_dir = Path(cwd, "coursework_c", "outputs", "temp")

    


    for filepath in files_to_check:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            print(f"\nExtracting {filepath.name}...\n")
            zip_ref.extractall(temp_dir)
            mat_data = [import_dataset(fp) for fp in temp_dir.iterdir()]
            print_data(mat_data)
