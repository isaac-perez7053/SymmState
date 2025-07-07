from typing import List, Tuple, Dict
import os
import copy
import numpy as np
from ordered_set import OrderedSet
from collections import Counter

from pymatgen.core import Structure, Element, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from symmstate.abinit import AbinitParser
from symmstate.utils.misc import Misc
from symmstate.pseudopotentials import PseudopotentialManager


class AbinitStructure(Structure):
    """
    Represent an Abinit-specific unit cell with Abinit file parameters and initialization routines.

    This class extends UnitCell by storing and managing variables specific to the Abinit program.
      - Parsing of an existing Abinit input file (abi_file) to extract cell parameters,
        coordinates, and atomic information.
      - Initialization from a Structure object to derive Abinit input parameters.
      - Optional incorporation of a symmetry-adapted basis using smodes_input and target_irrep.
      - Storage of parameters (e.g., acell, rprim, xred/xcart, and atomic elements) in self.vars.

    The initialization supports multiple input methods:
      1. If an abi_file is provided, the file is parsed (via AbinitParser), and the coordinates
         are set from the parsed data. If symmetry-adapted modes are also specified, they are used
         to initialize the parent UnitCell and update the unit cell parameters accordingly.
    """

    def __init__(self, vars: Dict) -> "AbinitStructure":
        """
        Initialize from a dictionary of Abinit variables.

        Parameters:
            vars (Dict): Dictionary containing all necessary Abinit variables

        Returns:
            AbinitStructure:
        """
        self.vars = vars
        AbinitParser.check(vars)

        lattice = Lattice(self.vars["rprim"] * self.vars["acell"])
        coords_are_cartesian = self.vars["coord_type"] == "xcart"

        super().__init__(
            lattice=lattice,
            species=self._convert_znucl_typat(),
            coords=self.vars["coordinates"],
            coords_are_cartesian=coords_are_cartesian,
        )

    @classmethod
    def from_file(cls, abi_file: str) -> "AbinitStructure":
        """
        Initialize from an Abinit input file.

        Parameters:
            abi_file (str): Filepath to Abinit input file

        Returns:
            AbinitStructure:
        """
        if not os.path.exists(abi_file):
            raise FileNotFoundError(f"The file {abi_file} does not exist!")
        vars = AbinitParser.parse_abinit_file(abi_file)
        return cls(vars)

    # Function is used when updating the file via SMODES supercell
    def update_unit_cell_parameters(self):
        """
        Update self.vars with current unit cell parameters.

        This method recalculates and updates parameters such as:
        - Total number of atoms (natom)
        - Sorted unique atomic numbers (znucl)
        - Atom type mapping (typat) and number of unique species (ntypat)
        - Number of bands (nband)
        - Reduced (xred) and cartesian (xcart) coordinates
        - Reorders pseudopotentials to correspond with the sorted atomic numbers.

        For reordering the pseudopotentials, each pseudo file (found in the directory at
        self.vars["pp_dirpath"] and with file names in self.vars["pseudos"]) is read to extract its
        atomic number. The pseudos are then ordered so that they match the order of sorted znucl.
        """
        # Get primitive vectors
        rprim = self.lattice.matrix

        # Get the original list of sites and extract species.
        sites = self.sites
        species = [site.specie for site in sites]

        natom = len(sites)
        # Preserve the original unique order as they appear.
        original_znucl = list(dict.fromkeys([s.Z for s in species]))  # e.g. [20, 8, 22]
        ntypat = len(original_znucl)

        # Calculate the number of bands using an external routine.
        nband = Misc.calculate_nband(self)

        # Get the original pseudos list (assumed to be set in self.vars).
        original_pseudos = self.vars.get("pseudos", [])
        if len(original_pseudos) != ntypat:
            raise ValueError(
                "The number of pseudopotentials does not match the number of unique atom types."
            )

        # Now sort the unique atomic numbers.
        sorted_znucl = sorted(original_znucl)  # e.g. [8, 20, 22]

        # Reorder the pseudopotentials to match sorted_znucl.
        self.vars["pp_dirpath"] = PseudopotentialManager().folder_path
        if not self.vars.get("pp_dirpath"):
            raise ValueError(
                "Pseudopotential directory (pp_dirpath) not found in self.vars."
            )

        # Build a mapping from pseudo file to its atomic number.
        pseudo_mapping = {}
        for pseudo in original_pseudos:
            # Remove any leading/trailing double quotes or whitespace from the pseudo filename.
            pseudo_clean = pseudo.strip('"').strip()
            filepath = os.path.join(self.vars.get("pp_dirpath"), pseudo_clean)
            try:
                with open(filepath, "r") as f:
                    lines = f.readlines()
                # Remove any double quotes from the second line before splitting
                tokens = lines[1].replace('"', "").split()
                if not tokens:
                    raise ValueError(
                        f"Pseudo file '{pseudo_clean}' has an invalid format."
                    )
                pseudo_z = float(tokens[0])
                pseudo_mapping[pseudo_clean] = int(round(pseudo_z))
            except Exception as e:
                raise ValueError(f"Error reading pseudo file '{pseudo_clean}': {e}")

        new_pseudos = []
        for z in sorted_znucl:
            matching = [pseudo for pseudo, pz in pseudo_mapping.items() if pz == z]
            if not matching:
                raise ValueError(
                    f"No pseudo file found corresponding to atomic number {z}."
                )
            # If multiple pseudo files are found for the same atomic number, the first one is chosen.
            new_pseudos.append(matching[0])

        # Recompute typat using the new sorted order.
        new_typat = [sorted_znucl.index(s.Z) + 1 for s in species]

        # Finally, update self.vars with the new parameters.
        self.vars.update(
            {
                "natom": natom,
                "rprim": rprim,
                "znucl": sorted_znucl,
                "typat": new_typat,
                "ntypat": ntypat,
                "nband": nband,
                "pseudos": new_pseudos,
                "xred": self.frac_coords,
                "xcart": self.cart_coords,
            }
        )

    def perturbation(
        self, perturbation: np.ndarray, coords_are_cartesian: bool = False
    ) -> "AbinitStructure":
        """
        Return a new AbinitStructure with perturbed coordinates.

        Parameters:
            perturbation (np.ndarray): The perturbation to apply.
            coords_are_cartesian (bool): If True, apply in cartesian; else, in fractional.

        Returns:
            AbinitStructure: New structure with perturbed coordinates.
        """
        perturbation = np.array(perturbation, dtype=float)

        # Decide which coordinates to perturb
        if coords_are_cartesian:
            coords = np.array(self.cart_coords)
            if perturbation.shape != coords.shape:
                raise ValueError(
                    "Perturbation must have the same shape as the cartesian coordinates."
                )
            new_cart = coords + perturbation
            # Convert back to fractional if needed
            new_frac = np.dot(new_cart, np.linalg.inv(self.lattice.matrix))
            new_coords = new_cart if self.vars["coord_type"] == "xcart" else new_frac
        else:
            coords = np.array(self.frac_coords)
            if perturbation.shape != coords.shape:
                raise ValueError(
                    "Perturbation must have the same shape as the fractional coordinates."
                )
            new_frac = coords + perturbation
            # Convert to cartesian if needed
            new_cart = np.dot(new_frac, self.lattice.matrix)
            new_coords = new_frac if self.vars["coord_type"] == "xred" else new_cart

        # Prepare new vars dict
        new_vars = self.abinit_parameters
        new_vars["coordinates"] = new_coords

        # Return new AbinitStructure
        return AbinitStructure(new_vars)

    def _convert_znucl_typat(self) -> List[str]:
        """
        Convert znucl/typat to element symbols.

        Parameters:
            None:

        Returns:
            List[str]: List of element symbols in order of typat

        """
        if "znucl" not in self.vars or "typat" not in self.vars:
            raise ValueError("Missing znucl or typat in Abinit file")

        znucl = self.vars["znucl"]
        typat = self.vars["typat"]
        return [Element.from_Z(znucl[t - 1]).symbol for t in typat]

    def typat_to_elem(self) -> List[str]:
        """
        Return a list of the unique elements of the system in the order maintained
        by the variable name typat

        Parameters:
            None:

        Returns:
            List[str]: List of element symbols in order of typat and are unique
        """

        znucl = self.vars["znucl"]
        typat = self.vars["typat"]
        return List(OrderedSet([Element.from_Z(znucl[t - 1]).symbol for t in typat]))

    def element_multiplicity(self) -> List[int]:
        """
        Return a list containing the number of each element where the
        order is specified by the typat variable

        Parameters:
            None:

        Returns:
            List[int]: List containing multiplicity of each element in order of typat

        """
        mult_typat = Counter(self.vars["typat"])
        return [count for _, count in mult_typat.items()]

    @property
    def abinit_parameters(self) -> dict:
        """Return copy of vars if available."""
        return self.vars.copy() if hasattr(self, "vars") else {}

    @property
    def copy(self) -> "AbinitStructure":
        """Return a copy of self."""
        return copy.deepcopy(self)

    @property
    def space_group(self) -> Tuple[int, str]:
        """Space group of the structure"""
        analyzer = SpacegroupAnalyzer(self)
        return (analyzer.get_space_group_number(), analyzer.get_space_group_symbol())

    def __repr__(self):
        def pretty_print_array(arr):
            arr = np.asarray(arr)
            if arr.ndim == 1:
                return "  " + "  ".join(
                    f"{x:.6g}" if isinstance(x, (int, float)) else str(x) for x in arr
                )
            elif arr.ndim == 2:
                return "\n    " + "\n    ".join(
                    "  ".join(
                        f"{x:.6g}" if isinstance(x, (int, float)) else str(x)
                        for x in row
                    )
                    for row in arr
                )
            else:
                return "\n" + "\n".join(
                    f"  Slice {i}:\n    "
                    + "\n    ".join(
                        "  ".join(
                            f"{x:.6g}" if isinstance(x, (int, float)) else str(x)
                            for x in row
                        )
                        for row in mat
                    )
                    for i, mat in enumerate(arr)
                )

        header = (
            "╭" + "─" * 40 + "╮\n"
            "│{:^40}│\n".format("AbinitStructure Summary") + "╰" + "─" * 40 + "╯"
        )

        lines = [header, ""]
        for key in sorted(self.vars):  # sorted for consistency
            value = self.vars[key]
            if isinstance(value, np.ndarray):
                pretty = pretty_print_array(value)
                lines.append(f"{key}:\n{pretty}")
            elif (
                isinstance(value, list)
                and value
                and all(isinstance(v, np.ndarray) for v in value)
            ):
                pretty = "\n    ".join(pretty_print_array(v) for v in value)
                lines.append(f"{key}:\n    {pretty}")
            else:
                lines.append(f"{key}: {value}")

        return "\n".join(lines)
