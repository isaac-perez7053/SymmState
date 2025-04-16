import re
from typing import Dict, List, Union
import numpy as np


class AbinitParser:
    """Parser for Abinit input files"""

    @staticmethod
    def parse_abinit_file(file_path: str) -> Dict:
        """Parse all parameters from Abinit file"""
        with open(file_path, "r") as f:
            content = f.read()

        # Determine coordinate type (xcart or xred)
        coord_type = None
        if AbinitParser._parse_matrix(content, "xcart", float) is not None:
            coord_type = "xcart"
        elif AbinitParser._parse_matrix(content, "xred", float) is not None:
            coord_type = "xred"

        # Parse all variables
        parsed_data = {
            "acell": AbinitParser._parse_array(content, "acell", float),
            "rprim": AbinitParser._parse_matrix(content, "rprim", float),
            coord_type: (
                AbinitParser._parse_matrix(content, coord_type, float)
                if coord_type
                else None
            ),
            "znucl": AbinitParser._parse_array(content, "znucl", int),
            "typat": AbinitParser._parse_array(content, "typat", int),
            "ecut": AbinitParser._parse_scalar(content, "ecut", int),
            "ecutsm": AbinitParser._parse_scalar(content, "ecutsm", float),
            "nshiftk": AbinitParser._parse_scalar(content, "nshiftk", int),
            "nband": AbinitParser._parse_scalar(content, "nband", int),
            "diemac": AbinitParser._parse_scalar(content, "diemac", float),
            "toldfe": AbinitParser._parse_scalar(content, "toldfe", float),
            "tolvrs": AbinitParser._parse_scalar(content, "tolvrs", float),
            "tolsym": AbinitParser._parse_scalar(content, "tolsym", float),
            "ixc": AbinitParser._parse_scalar(content, "ixc", int),
            "kptrlatt": AbinitParser._parse_matrix(content, "kptrlatt", int),
            "pp_dirpath": AbinitParser._parse_string(content, "pp_dirpath"),
            "pseudos": AbinitParser._parse_array(content, "pseudos", str),
            "natom": AbinitParser._parse_scalar(content, "natom", int),
            "ntypat": AbinitParser._parse_scalar(content, "ntypat", int),
            "kptopt": AbinitParser._parse_scalar(content, "kptopt", int),
            "chkprim": AbinitParser._parse_scalar(content, "chkprim", int),
            "shiftk": AbinitParser._parse_array(content, "shiftk", float),
            "nstep": AbinitParser._parse_scalar(content, "nstep", int),
            "useylm": AbinitParser._parse_scalar(content, "useylm", int),
        }

        # Determine the type of convergence criteria used
        init_methods = [
            parsed_data["toldfe"],
            parsed_data["tolvrs"],
            parsed_data["tolsym"],
        ]
        if sum(x is not None for x in init_methods) != 1:
            raise ValueError("Specify exactly one convergence criteria")

        conv_criteria = None
        if parsed_data["toldfe"] is not None:
            conv_criteria = "toldfe"
        elif parsed_data["tolsym"] is not None:
            conv_criteria = "tolsym"
        elif parsed_data["tolvrs"] is not None:
            conv_criteria = "tolvrs"

        if conv_criteria is None:
            raise ValueError("Please specify a convergence criteria")
        parsed_data["conv_criteria"] = conv_criteria

        # Remove None values
        return {k: v for k, v in parsed_data.items() if v is not None}

    @staticmethod
    def _parse_array(content: str, param_name: str, dtype: type) -> Union[List, None]:
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

    @staticmethod
    def _parse_matrix(content: str, key: str, dtype: type) -> Union[np.ndarray, None]:
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
    def _parse_scalar(content: str, key: str, dtype: type) -> Union[type, None]:
        match = re.search(rf"{key}\s+([\d\.+-dDeE]+)", content)
        if match:
            # Replace 'd' or 'D' with 'e' for compatibility with Python floats
            value = match.group(1).replace("d", "e").replace("D", "e")
            return dtype(value)
        return None

    @staticmethod
    def _parse_string(content: str, key: str) -> Union[str, None]:
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
