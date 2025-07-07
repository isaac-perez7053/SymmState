from typing import Dict
import numpy as np

from symmstate.utils import DataParser


class AbinitParser:
    """Parser for Abinit input files"""

    def __init__(self):
        pass

    @staticmethod
    def parse_abinit_file(file_path: str) -> Dict:
        """
        Parse all parameters from Abinit file

        Parameters:
            file_path (str): A string that represents the abinit
                input file

        Returns:
            Dict: Dictionary containing all extract Abinit input variables
        """

        with open(file_path, "r") as f:
            content = f.read()

        # Remove the comments in the file
        content = DataParser.remove_comments(content)

        # Determine coordinate type (xcart or xred)
        coord_type = None
        if DataParser.parse_matrix(content, "xcart", dtype=np.float64) is not None:
            coord_type = "xcart"
        elif DataParser.parse_matrix(content, "xred", dtype=np.float64) is not None:
            coord_type = "xred"
        else:
            raise ValueError(
                "Missing atomic coordinates: expected 'xred' or 'xcart' in input."
            )

        # Parse all variables
        parsed_data = {
            "acell": DataParser.parse_array(content, "acell", float),
            "rprim": DataParser.parse_matrix(content, "rprim", float),
            "coord_type": coord_type,
            "coordinates": DataParser.parse_matrix(content, coord_type, float),
            "znucl": DataParser.parse_array(content, "znucl", int),
            "typat": DataParser.parse_array(content, "typat", int),
            "ecut": DataParser.parse_scalar(content, "ecut", float),
            "ecutsm": DataParser.parse_scalar(content, "ecutsm", float),
            "nshiftk": DataParser.parse_scalar(content, "nshiftk", int),
            "nband": DataParser.parse_scalar(content, "nband", int),
            "diemac": DataParser.parse_scalar(content, "diemac", float),
            "toldfe": DataParser.parse_scalar(content, "toldfe", float),
            "tolvrs": DataParser.parse_scalar(content, "tolvrs", float),
            "tolsym": DataParser.parse_scalar(content, "tolsym", float),
            "ixc": DataParser.parse_scalar(content, "ixc", int),
            "kptrlatt": DataParser.parse_matrix(content, "kptrlatt", int),
            "pp_dirpath": DataParser.parse_string(content, "pp_dirpath"),
            "pseudos": DataParser.parse_array(content, "pseudos", str),
            "natom": DataParser.parse_scalar(content, "natom", int),
            "ntypat": DataParser.parse_scalar(content, "ntypat", int),
            "kptopt": DataParser.parse_scalar(content, "kptopt", int),
            "chkprim": DataParser.parse_scalar(content, "chkprim", int),
            "shiftk": [
                np.array(row)
                for row in DataParser.parse_matrix(content, "shiftk", float).tolist()
            ],  # shiftk is now a list of lists
            "nstep": DataParser.parse_scalar(content, "nstep", int),
            "useylm": DataParser.parse_scalar(content, "useylm", int),
            "ngkpt": DataParser.parse_array(content, "ngkpt", float),
        }

        # Run a series of checks to make sure the data is valid
        AbinitParser.check(parsed_data)

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

        return parsed_data

    @staticmethod
    def check(parsed_data: Dict) -> None:
        """
        Check if the parsed data is valid

        Parameters:
            parsed_data (Dict): The dictionary containing the extracted variables

        Returns:
            None:
        """

        if (parsed_data["ngkpt"] is not None) and (parsed_data["kptrlatt"] is not None):
            raise ValueError("ngkpt and kptrlatt cannot both be defined")

        required_keys = [
            "acell",
            "coord_type",
            "coordinates",
            "znucl",
            "typat",
            "ixc",
            "pseudos",
            "natom",
            "ntypat",
        ]

        for key in required_keys:
            if parsed_data[key] is None:
                raise ValueError(f"Missing required variable: {key}")

        # Check coordinates shape
        coords = parsed_data["coordinates"]
        natom = parsed_data["natom"]
        if not (isinstance(coords, np.ndarray) and coords.shape == (natom, 3)):
            raise ValueError(
                f"Coordinates array must have shape ({natom}, 3), got {coords.shape}"
            )

        # Check ntypat matches unique typat
        typat = parsed_data["typat"]
        ntypat = parsed_data["ntypat"]
        unique_types = len(set(typat))
        if ntypat != unique_types:
            raise ValueError(
                f"ntypat ({ntypat}) does not match number of unique types in typat ({unique_types})"
            )

        # Check length of pseudos matches ntypat
        pseudos = parsed_data["pseudos"]
        if len(pseudos) != ntypat:
            raise ValueError(
                f"Length of pseudos ({len(pseudos)}) does not match ntypat ({ntypat})"
            )

        # Determine the type of convergence criteria used
        init_methods = [
            parsed_data["toldfe"],
            parsed_data["tolvrs"],
            parsed_data["tolsym"],
        ]
        if sum(x is not None for x in init_methods) != 1:
            raise ValueError("Specify exactly one convergence criteria")

    @staticmethod
    def abinit_variable_descriptions():
        """Print descriptions of all Abinit variables currently supported"""
        return {
            "acell": "Lattice vectors in Angstrom (Tuple[float, float, float])",
            "rprim": "Primitive lattice vectors in Angstrom (Matrix[3, 3])",
            "xcart": "Cartesian coordinates of atoms in Angstrom (Matrix[natom, 3])",
            "xred": "Reduced coordinates of atoms (Matrix[natom, 3])",
            "znucl": "Nuclear charges of atoms (arr[ntypat])",
            "typat": "Type of each atom (arr[natom])",
            "ecut": "Plane-wave cutoff energy in Hartree (float)",
            "ecutsm": "Smearing energy for Fermi-Dirac smearing (float)",
            "nshiftk": "Number of k-point shifts (Tuple[float, float, float])",
            "nband": "Number of bands (int)",
            "diemac": "Macroscopic dielectric constant (float)",
            "toldfe": "Energy convergence criterion (float)",
            "tolvrs": "Residual convergence criterion (float)",
            "tolsym": "Symmetry convergence criterion (float)",
            "ixc": "Exchange-correlation functional index (int)",
            "kptrlatt": "K-point lattice vectors (Matrix)",
            "pp_dirpath": "Path to pseudopotential directory (str)",
            "pseudos": "List of pseudopotentials for each atom type (arr[str] * ntypat)",
            "natom": "Total number of atoms in the structure (int)",
            "ntypat": "Number of different atom types (int)",
            "kptopt": "K-point generation option (int)",
            "chkprim": "Check primitive cell option (0 or 1) (int)",
            "shiftk": "K-point shifts for symmetry operations (Tuple[float, float, float])",
            "nstep": "Number of steps for the calculation (int)",
            "useylm": "'1' to use spherical harmonics, '0' otherwise (int)",
            "ngkpt": "'ngkpt' is a list of integers defining the number of k-points (in the x, y, z directions) in the reciprocal space grid. (arr[int, int, int])",
        }

    @staticmethod
    def is_supported(key: str) -> bool:
        """
        Check if a variable is supported by the parser

        Parameters:
            key (str): The queried key

        Returns:
            bool: true if supported and false otherwise
        """
        return key in AbinitParser.abinit_variable_descriptions()

    def __str__(self):
        return "\n".join(
            f"{name}: {desc}"
            for name, desc in AbinitParser.abinit_variable_descriptions().items()
        )
