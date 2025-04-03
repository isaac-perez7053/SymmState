import os
import shutil
from pathlib import Path
from typing import Union

def get_unique_filename(base_name: Union[str, Path], directory: str = ".") -> str:
    """Generate unique filename with incremental suffix if conflicts exist"""
    base_path = Path(directory) / base_name
    if not base_path.exists():
        return str(base_path)

    counter = 1
    while True:
        new_name = f"{base_path.stem}_{counter:03}{base_path.suffix}"
        candidate = base_path.with_name(new_name)
        if not candidate.exists():
            return str(candidate)
        counter += 1

def safe_file_copy(src: Union[str, Path], dest_dir: Union[str, Path]) -> Path:
    """Copy file with conflict resolution"""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(src).name
    dest_path = Path(get_unique_filename(dest_path))
    shutil.copy(src, dest_path)
    return dest_path