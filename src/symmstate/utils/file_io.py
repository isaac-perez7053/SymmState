import os
import shutil
from pathlib import Path
from typing import Union


def safe_file_copy(src: Union[str, Path], dest_dir: Union[str, Path]) -> Path:
    """Copy file with conflict resolution"""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(src).name
    dest_path = Path(get_unique_filename(dest_path))
    shutil.copy(src, dest_path)
    return dest_path


def get_unique_filename(base_name, directory=".") -> str:
    """
    Get the unique filename of a file
    """
    # Get the full path for the base file
    full_path = os.path.join(directory, base_name)

    # If the file does not exist, return the base name
    if not os.path.isfile(full_path):
        return base_name

    # Otherwise, start creating new filenames with incrementing numbers
    counter = 0
    while True:
        # Format the new filename with leading zeros
        new_name = f"{os.path.splitext(base_name)[0]}_{counter:03}{os.path.splitext(base_name)[1]}"
        new_full_path = os.path.join(directory, new_name)

        # Check if the new filename is available
        if not os.path.isfile(new_full_path):
            return new_name

        # Increment the counter
        counter += 1
