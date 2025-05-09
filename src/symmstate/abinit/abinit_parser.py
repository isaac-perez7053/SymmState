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
            "ngkpt": DataParser.parse_array(content, "ngkpt", float)
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


