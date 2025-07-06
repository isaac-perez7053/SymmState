"""
The DataParser class contains functions related to extracting data from a file and 
putting them in useful structures.
"""

from typing import Optional
import numpy as np
import re
from typing import Union, List


class DataParser:
    def __init__(self):
        pass

    @staticmethod
    def remove_comments(content: str) -> str:
        """Removes comments beginning with #"""
        return "\n".join(line.split("#", 1)[0] for line in content.splitlines())

    @staticmethod
    def grab_energy(abo_file: str) -> None:
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
                print("Total energy not found.")
        except FileNotFoundError:
            print(f"The file {abo_file} was not found.")
        return energy

    @staticmethod
    def grab_flexo_tensor(anaddb_file: str) -> None:
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
            print(f"The file {anaddb_file} was not found.")
        return flexo_tensor

    @staticmethod
    def grab_piezo_tensor(anaddb_file: str) -> None:
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
            print(f"The file {anaddb_file} was not found.")
        return piezo_tensor_clamped, piezo_tensor_relaxed

    @staticmethod
    def parse_matrix(content: str, key: str, dtype: type) -> Union[np.ndarray, None]:
        lines = content.strip().splitlines()
        matrices: List[List] = []
        found_key = False
        start_index = -1

        for i, line in enumerate(lines):
            if re.search(rf"\b{re.escape(key)}\b", line):
                found_key = True
                tokens = line.split()
                if len(tokens) > 1 and re.match(r"^[-+.\deEdD]", tokens[1]):
                    # Matrix starts on the same line
                    try:
                        row = [
                            dtype(tok.replace("d", "e").replace("D", "e"))
                            for tok in tokens[1:]
                        ]
                        matrices.append(row)
                    except ValueError:
                        pass
                    start_index = i + 1
                else:
                    start_index = i + 1
                break

        if not found_key or start_index < 0:
            return None

        for j in range(start_index, len(lines)):
            line = lines[j].strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) != 3:
                break
            try:
                row = [dtype(tok.replace("d", "e").replace("D", "e")) for tok in tokens]
                matrices.append(row)
            except ValueError:
                break

        return np.array(matrices, dtype=dtype) if matrices else None

    @staticmethod
    def parse_scalar(
        content: str, key: str, dtype: type, all_matches: bool = False
    ) -> Union[type, List[type], None]:
        """
        Extract scalar value(s) following `key`.
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
                # skip anything that still fails conversion ;)
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

    @staticmethod
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

        if all_matches != False:
            for result in results:
                result = np.array(result)
        else:
            results[0] = np.array(results[0])

        return results if all_matches else results[0]
