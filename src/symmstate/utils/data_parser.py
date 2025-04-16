from typing import Optional
import numpy as np
import re
import logging


class DataParser:
    def __init__(self):
        pass

    @staticmethod
    def grab_energy(abo_file: str, logger: logging = None) -> None:
        """
        Retrieves the total energy from a specified Abinit output file.
        """
        energy = None
        if abo_file is None:
            raise Exception("Please specify the abo file you are attempting to access")
        total_energy_value: Optional[str] = None
        try:
            with open(abo_file) as f:
                abo_content: str = f.read()
            match = re.search(r"total_energy\s*:\s*(-?\d+\.\d+E?[+-]?\d*)", abo_content)
            if match:
                total_energy_value = match.group(1)
                energy: float = float(total_energy_value)
            else:
                (logger.info if logger is not None else print)(
                    "Total energy not found.", logger=logger
                )
        except FileNotFoundError:
            (logger.info if logger is not None else print)(
                f"The file {abo_file} was not found.", logger=logger
            )
        return energy

    @staticmethod
    def grab_flexo_tensor(anaddb_file: str, logger: logging = None) -> None:
        """
        Retrieves the TOTAL flexoelectric tensor from the specified file.
        """
        flexo_tensor: Optional[np.ndarray] = None
        try:
            with open(anaddb_file) as f:
                abo_content: str = f.read()
            flexo_match = re.search(
                r"TOTAL flexoelectric tensor \(units= nC/m\)\s*\n\s+xx\s+yy\s+zz\s+yz\s+xz\s+xy\n((?:.*\n){9})",
                abo_content,
            )
            if flexo_match:
                tensor_strings = flexo_match.group(1).strip().split("\n")
                flexo_tensor = np.array(
                    [list(map(float, line.split()[1:])) for line in tensor_strings]
                )
        except FileNotFoundError:
            (logger.info if logger is not None else print)(
                f"The file {anaddb_file} was not found.", logger=logger
            )
        return flexo_tensor

    @staticmethod
    def parse_tensor(tensor_str: str, logger: logging = None) -> np.ndarray:
        """
        Parses a tensor string into a NumPy array.
        """
        lines = tensor_str.strip().splitlines()
        tensor_data = []
        for line in lines:
            elements = line.split()
            if all(part.lstrip("-").replace(".", "", 1).isdigit() for part in elements):
                try:
                    numbers = [float(value) for value in elements]
                    tensor_data.append(numbers)
                except ValueError as e:
                    (logger.info if logger is not None else print)(
                        f"Could not convert line to numbers: {line}, Error: {e}",
                        logger=logger,
                    )
                    raise
        return np.array(tensor_data)

    @staticmethod
    def grab_piezo_tensor(anaddb_file: str, logger: logging = None) -> None:
        """
        Retrieves the clamped and relaxed ion piezoelectric tensors.
        """
        piezo_tensor_clamped: Optional[np.ndarray] = None
        piezo_tensor_relaxed: Optional[np.ndarray] = None
        try:
            with open(anaddb_file) as f:
                abo_content: str = f.read()
            clamped_match = re.search(
                r"Proper piezoelectric constants \(clamped ion\) \(unit:c/m\^2\)\s*\n((?:\s*-?\d+\.\d+\s+\n?)+)",
                abo_content,
            )
            if clamped_match:
                clamped_strings = clamped_match.group(1).strip().split("\n")
                piezo_tensor_clamped = np.array(
                    [list(map(float, line.split())) for line in clamped_strings]
                )
            relaxed_match = re.search(
                r"Proper piezoelectric constants \(relaxed ion\) \(unit:c/m\^2\)\s*\n((?:\s*-?\d+\.\d+\s+\n?)+)",
                abo_content,
            )
            if relaxed_match:
                relaxed_strings = relaxed_match.group(1).strip().split("\n")
                piezo_tensor_relaxed = np.array(
                    [list(map(float, line.split())) for line in relaxed_strings]
                )
        except FileNotFoundError:
            (logger.info if logger is not None else print)(
                f"The file {anaddb_file} was not found.", logger=logger
            )
        return piezo_tensor_clamped, piezo_tensor_relaxed
