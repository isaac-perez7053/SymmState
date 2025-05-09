from typing import Optional
import numpy as np
import re
import logging
from typing import Union, List


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
    

    @staticmethod
    def parse_matrix(content: str, key: str, dtype: type) -> Union[np.ndarray, None]:
        """Improved matrix parsing that allows negative numbers.

        Searches for a line starting with the key and then reads subsequent lines
        that start with either a digit or a minus sign.
        """
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if re.fullmatch(rf"\s*{key}\s*", line):
                matrix = []
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip()
                    # Allow lines starting with '-' or a digit.
                    if not next_line or not re.match(r"^[-\d]", next_line):
                        break
                    matrix.append([dtype(x) for x in next_line.split()])
                return np.array(matrix) if matrix else None
        return None

    @staticmethod
    def parse_scalar(content: str, key: str, dtype: type) -> Union[type, None]:
        match = re.search(rf"{key}\s+([\d\.+-dDeE]+)", content)
        if match:
            # Replace 'd' or 'D' with 'e' for compatibility with Python floats
            value = match.group(1).replace("d", "e").replace("D", "e")
            return dtype(value)
        return None

    @staticmethod
    def parse_string(content: str, key: str) -> Union[str, None]:
        """
        Parse a string value from content corresponding to a given key.

        The function searches for the key followed by one or more spaces and then a
        double-quoted string, and returns the extracted string. If the key is not found,
        or if a quoted string is not present, the function returns None.

        **Parameters:**
            key (str): The key to search for in the content.
            content (str): The string content to parse.

        **Returns:**
            Union[str, None]: The extracted string value (without quotes) if found, otherwise None.
        """
        match = re.search(rf'{key}\s+"([^"]+)"', content)
        if match:
            return match.group(1)
        return None
    
    @staticmethod
    def parse_array(content: str, param_name: str, dtype: type) -> Union[List, None]:
        """Parse values for a given parameter name with specified data type.

        Handles multiplicity like 'param_name 1 2 1.0*3' or 'key 4 5 6.2*2'
        and returns expanded list with elements converted to given dtype.
        """
        regex_pattern = rf"^{param_name}\s+([^\n]+)"
        match = re.search(regex_pattern, content, re.MULTILINE)

        if not match:
            return None

        tokens = match.group(1).replace(",", " ").split()
        result = []

        for token in tokens:
            if "*" in token:
                parts = token.split("*")
                if len(parts) == 2:
                    # Switch order: the left side is the count and the right is the value.
                    count_str, val = parts
                    try:
                        count = int(count_str)
                    except ValueError:
                        count = int(
                            float(count_str)
                        )  # Handle case where count may be a float
                    result.extend([dtype(val)] * count)
            else:
                result.append(dtype(token))

        return result

