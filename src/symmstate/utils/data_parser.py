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
                    "Total energy not found."
                )
        except FileNotFoundError:
            (logger.info if logger is not None else print)(
                f"The file {abo_file} was not found."
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
    def parse_matrix(
        content: str, key: str, dtype: type, all_matches: bool = False
    ) -> Union[np.ndarray, List[np.ndarray], None]:
        """
        Extract one or more matrices that follow lines containing exactly `key`.

        A matrix is defined as the block of subsequent non-empty lines
        each starting with a digit or '-' (allowing negative numbers).

        By default returns the first match as an np.ndarray.
        If all_matches=True, returns a list of np.ndarrays.
        Returns None if no matching key line or no data.
        """
        # Pattern to find lines that are exactly `key` (with optional indent)
        key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*$", flags=re.MULTILINE)
        lines = content.splitlines()

        matrices: List[np.ndarray] = []
        # Iterate over every line that matches the key
        for match in key_pattern.finditer(content):
            # figure out which line number it was
            start_line = content[: match.start()].count("\n")
            block: List[List] = []
            # collect following lines that start with digit or '-'
            for ln in lines[start_line + 1 :]:
                stripped = ln.strip()
                if not stripped or not re.match(r"^[-\d]", stripped):
                    break
                # convert each token in that row
                row = [dtype(tok) for tok in stripped.split()]
                block.append(row)
            if block:
                matrices.append(np.array(block, dtype=dtype))

        if not matrices:
            return None

        return matrices if all_matches else matrices[0]

    @staticmethod
    def parse_scalar(
        content: str, key: str, dtype: type, all_matches: bool = False
    ) -> Union[type, List[type], None]:
        """
        Extract scalar value(s) following `key`.

        - Recognizes FORTRAN‑style exponents (d/D) and converts them to 'e' for Python.
        - By default returns the first match as a single dtype.
        - If all_matches=True, returns List[dtype], one element per occurrence.
        - Returns None if no numeric match is found.
        """
        # Regex for a number with optional sign, decimal part, and e/D exponent
        num_re = (
            r"[+-]?"  # optional sign
            r"\d+(?:\.\d*)?"  # digits, optional fractional part
            r"(?:[eEdD][+-]?\d+)?"  # optional exponent with e/E or d/D
        )

        # allow indent, then key, whitespace, then capture the number
        pattern = rf"^\s*{re.escape(key)}\s+({num_re})"

        # find all occurrences
        raw_matches = [
            m.group(1) for m in re.finditer(pattern, content, flags=re.MULTILINE)
        ]
        if not raw_matches:
            return None

        # normalize and convert
        converted: List[type] = []
        for raw in raw_matches:
            norm = raw.replace("d", "e").replace("D", "e")
            try:
                converted.append(dtype(norm))
            except ValueError:
                # skip anything that still fails conversion
                pass

        if not converted:
            return None

        return converted if all_matches else converted[0]

    @staticmethod
    def parse_string(
        content: str, key: str, all_matches: bool = False
    ) -> Union[str, List[str], None]:
        """
        Extract the double‑quoted string(s) following `key`.

        By default returns the first match as a string.
        If all_matches=True, returns a list of all matched strings.
        Returns None if there are no matches.
        """
        # allow optional indent before the key, then key, whitespace, then "…"
        pattern = rf'^\s*{re.escape(key)}\s+"([^"]+)"'

        # collect all matches
        results: List[str] = [
            m.group(1) for m in re.finditer(pattern, content, flags=re.MULTILINE)
        ]

        if not results:
            return None

        return results if all_matches else results[0]

    def parse_array(
        content: str, param_name: str, dtype: type, all_matches: bool = False
    ) -> Union[List, List[List], None]:
        """
        Parse the line(s) starting with `param_name`.

        If dtype is int/float, only numeric tokens (and multiplicities) are captured.
        If dtype is str, all tokens on the line are captured.

        Multiplicity tokens look like "3*1.23" and expand to [1.23, 1.23, 1.23].

        By default returns the first match as a List.
        If all_matches=True, returns List[List] (one sublist per match).
        Returns None if there are no matches.
        """
        # float‑literal regex with no capturing subgroups
        float_re = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

        if dtype is str:
            # grab everything after param_name (incl. units)
            pattern = rf"^\s*{param_name}\s+([^\n]+)"
        else:
            # grab only floats and multiplicity patterns
            # but we don't enforce multiplicity in the regex; we'll handle it below
            pattern = rf"^\s*{param_name}\s+(.+)"

        # find all occurrences
        results: List[List] = []
        for m in re.finditer(pattern, content, flags=re.MULTILINE):
            line = m.group(1).strip()
            tokens = line.replace(",", " ").split()
            row: List = []

            for tok in tokens:
                if dtype is str:
                    # raw string mode
                    row.append(tok)
                else:
                    # numeric mode: handle multiplicity tokens
                    if "*" in tok:
                        left, right = tok.split("*", 1)
                        try:
                            count = int(left)
                        except ValueError:
                            # maybe "1.0*val"—cast float to int
                            count = int(float(left))
                        # normalize any 'd' exponents
                        val_str = right.replace("d", "e").replace("D", "e")
                        try:
                            val = dtype(val_str)
                        except ValueError:
                            continue
                        row.extend([val] * count)
                    else:
                        # plain numeric token
                        val_str = tok.replace("d", "e").replace("D", "e")
                        try:
                            row.append(dtype(val_str))
                        except ValueError:
                            continue

            if row:
                results.append(row)

        if not results:
            return None

        return results if all_matches else results[0]
