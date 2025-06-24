from symmstate.utils import DataParser
from typing import Dict


class AbinitParser:
    """Parser for Abinit input files"""

    @staticmethod
    def parse_abinit_file(file_path: str) -> Dict:
        """Parse all parameters from Abinit file"""
        with open(file_path, "r") as f:
            content = f.read()

        # Determine coordinate type (xcart or xred)
        coord_type = None
        if DataParser.parse_matrix(content, "xcart", float) is not None:
            coord_type = "xcart"
        elif DataParser.parse_matrix(content, "xred", float) is not None:
            coord_type = "xred"

        # Parse all variables
        parsed_data = {
            "acell": DataParser.parse_array(content, "acell", float),
            "rprim": DataParser.parse_matrix(content, "rprim", float),
            coord_type: (
                DataParser.parse_matrix(content, coord_type, float)
                if coord_type
                else None
            ),
            "znucl": DataParser.parse_array(content, "znucl", int),
            "typat": DataParser.parse_array(content, "typat", int),
            "ecut": DataParser.parse_scalar(content, "ecut", int),
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
            "shiftk": DataParser.parse_array(content, "shiftk", float),
            "nstep": DataParser.parse_scalar(content, "nstep", int),
            "useylm": DataParser.parse_scalar(content, "useylm", int),
            "ngkpt": DataParser.parse_array(content, "ngkpt", float),
        }

        # Check if all required keys are present
        required_keys = [
            "acell",
            coord_type,
            "znucl",
            "typat",
            "ecut",
            "ixc",
            "pseudos",
            "natom",
            "ntypat",
        ]
        for key in required_keys:
            if key not in parsed_data or parsed_data[key] is None:
                raise ValueError(f"Missing required variable: {key}")

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

        return {k: v for k, v in parsed_data.items()}

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
        """Check if a variable is supported by the parser"""
        return key in AbinitParser.abinit_variable_descriptions()

    def __str__(self):
        return "\n".join(
            f"{name}: {desc}"
            for name, desc in AbinitParser.abinit_variable_descriptions().items()
        )
