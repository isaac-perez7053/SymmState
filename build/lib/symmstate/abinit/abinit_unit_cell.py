"""
Abinit Unit Cell module.

This module implements the AbinitUnitCell class, which extends UnitCell with Abinit-specific functionality.
It stores and manages parameters parsed from an Abinit file or derived from a Structure object.
"""

import copy
from typing import Optional, List
import numpy as np
import os

from pymatgen.core import Structure, Element

from symmstate.unit_cell_module import UnitCell
from symmstate.utils.abinit_parser import AbinitParser
from symmstate.utils.misc import Misc
from symmstate.pseudopotentials import PseudopotentialManager


class AbinitUnitCell(UnitCell):
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
      2. If a Structure object is provided, the Abinit parameters are derived directly from the unit cell.
      3. If neither input is valid, an error is raised.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        abi_file: Optional[str] = None,
        unit_cell: Optional[Structure] = None,
        *,
        smodes_input: Optional[str] = None,
        target_irrep: Optional[str] = None,
        symm_prec: float = 1e-5,
    ):
        """
        Initialize an AbinitUnitCell instance.

        Parameters:
            abi_file (Optional[str]):
                Path to an existing Abinit input file. If provided, it is parsed to populate
                the unit cell parameters (e.g., acell, rprim, coordinates, and atomic species).
            unit_cell (Optional[Structure]):
                A Structure object used to generate Abinit parameters when no abi_file is provided.
            smodes_input (Optional[str], keyword-only):
                Input data for symmetry-adapted modes, which defines the symmetry-adapted basis.
            target_irrep (Optional[str], keyword-only):
                The target irreducible representation corresponding to the symmetry-adapted basis.
            symm_prec (float, optional):
                The precision tolerance for symmetry operations. Default is 1e-5.

        Raises:
            TypeError: If the provided unit_cell is not a Structure.
            ValueError: If neither an abi_file nor a valid unit_cell is provided, or if the parsed
                        Abinit file lacks coordinate information.
        """
        self.vars = {}  # Initialize empty dictionary first

        if abi_file:
            self.abi_file = abi_file
            self.vars = AbinitParser.parse_abinit_file(abi_file)

            # Handle coordinates properly
            if "xred" in self.vars:
                coordinates = self.vars["xred"]
                coords_are_cartesian = False
            elif "xcart" in self.vars:
                coordinates = self.vars["xcart"]
                coords_are_cartesian = True
            else:
                raise ValueError("No coordinates found in Abinit file")

            # Initialize parent class with parsed parameters
            if (smodes_input is not None) and (target_irrep is not None):
                super().__init__(
                    smodes_file=smodes_input,
                    target_irrep=target_irrep,
                    symm_prec=symm_prec,
                )
                # Update parameters with new cell
                self.update_unit_cell_parameters()

            else:
                super().__init__(
                    acell=self.vars["acell"],
                    rprim=self.vars["rprim"],
                    coordinates=coordinates,
                    coords_are_cartesian=coords_are_cartesian,
                    elements=self._convert_znucl_typat(),
                )

                # Ensure instance has both xcart and xred
                if "xcart" not in self.vars:
                    self.vars["xcart"] = self.grab_cartesian_coordinates()
                elif "xred" not in self.vars:
                    self.vars["xred"] = self.grab_reduced_coordinates()

        elif unit_cell:
            if not isinstance(unit_cell, Structure):
                raise TypeError("unit_cell must be a Structure")
            super().__init__(structure=unit_cell)
            self._derive_abinit_parameters()
        else:
            raise ValueError("Provide a valid input for initailization")

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
        rprim = self.grab_primitive_vectors()

        # Get the original list of sites and extract species.
        sites = self.structure.sites
        species = [site.specie for site in sites]

        natom = len(sites)
        # Preserve the original unique order as they appear.
        original_znucl = list(dict.fromkeys([s.Z for s in species]))  # e.g. [20, 8, 22]
        ntypat = len(original_znucl)

        # Calculate the number of bands using an external routine.
        nband = Misc.calculate_nband(self.structure)

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
                "xred": self.grab_reduced_coordinates(),
                "xcart": self.grab_cartesian_coordinates(),
            }
        )

    def _derive_abinit_parameters(self):
        """Derive parameters and populate vars."""
        rprim = self.grab_primitive_vectors()
        natom = len(self.structure)
        znucl = sorted({e.Z for e in self.structure.species})
        typat = [znucl.index(s.Z) + 1 for s in self.structure.species]
        ntypat = len(znucl)
        xred = np.array(self.structure.frac_coords)
        xcart = np.array(self.structure.cart_coords)

        # Update vars
        self.vars.update(
            {
                "acell": self.structure.lattice.abc,
                "rprim": np.array(rprim),
                "znucl": znucl,
                "typat": typat,
                "natom": natom,
                "ntypat": ntypat,
                "xred": xred,
                "xcart": xcart,
            }
        )

    def _init_from_abinit_vars(self):
        """Initialize from parsed Abinit variables."""
        # Extract critical structural parameters
        acell = self.vars["acell"]
        rprim = self.vars["rprim"]

        # Handle coordinate system
        if "xcart" in self.vars:
            coordinates = self.vars["xcart"]
            coords_are_cartesian = True
        elif "xred" in self.vars:
            coordinates = self.vars["xred"]
            coords_are_cartesian = False
        else:
            raise ValueError("No atomic coordinates found in Abinit file")

        # Convert znucl/typat to element symbols
        elements = self._convert_znucl_typat()

        # Initialize parent class
        super().__init__(
            acell=acell,
            rprim=rprim,
            coordinates=coordinates,
            coords_are_cartesian=coords_are_cartesian,
            elements=elements,
        )

    def _convert_znucl_typat(self) -> List[str]:
        """Convert znucl/typat to element symbols."""
        if "znucl" not in self.vars or "typat" not in self.vars:
            raise ValueError("Missing znucl or typat in Abinit file")

        znucl = self.vars["znucl"]
        typat = self.vars["typat"]
        return [Element.from_Z(znucl[t - 1]).symbol for t in typat]

    @property
    def abinit_parameters(self) -> dict:
        """Return copy of vars if available."""
        return self.vars.copy() if hasattr(self, "vars") else {}

    # --------------------------
    # Initialization Methods
    # --------------------------

    @staticmethod
    def _process_atoms(atom_list):
        """Process a list of atoms into Abinit parameters."""
        # Calculate the total number of atoms
        num_atoms = len(atom_list)

        # Get the unique elements and their respective indices
        unique_elements = list(dict.fromkeys(atom_list))
        element_index = {element: i + 1 for i, element in enumerate(unique_elements)}

        # Create typat list based on unique elements' indices
        typat = [element_index[element] for element in atom_list]

        # Create znucl list with atomic numbers using pymatgen
        znucl = [Element(el).Z for el in unique_elements]

        return num_atoms, len(unique_elements), typat, znucl

    # --------------------------
    # Utilities
    # --------------------------

    def copy_abinit_unit_cell(self):
        """
        Create a deep copy of the current AbinitUnitCell instance.

        Returns:
            AbinitUnitCell: A new instance that is a deep copy of the current instance.
        """
        # Perform a deep copy to ensure all nested objects are also copied
        return copy.deepcopy(self)

    def change_coordinates(
        self, new_coordinates: np.ndarray, coords_are_cartesian: bool = False
    ):
        """
        Update the structure's coordinates with a new set of values.

        This method replaces the current atomic coordinates in the structure without calling
        the superclass method. It creates a new Structure object with the provided coordinates
        and updates internal variables for both cartesian and reduced coordinates.

        Parameters:
            new_coordinates (np.ndarray):
                A NumPy array containing the new atomic coordinates. Its shape must match the
                current reduced coordinate array stored in self.vars['xred'].
            coords_are_cartesian (bool, optional):
                Flag indicating whether the new coordinates are given in cartesian form. The default
                value is False (i.e., coordinates are assumed to be in reduced form).

        Raises:
            TypeError:
                If new_coordinates is not a NumPy array.
            ValueError:
                If the shape of new_coordinates does not match the shape of self.vars['xred'].
        """
        if not isinstance(new_coordinates, np.ndarray):
            raise TypeError("Ensure that the new coordinates are a numpy array")

        if new_coordinates.shape != self.vars["xred"].shape:
            raise ValueError("Ensure that the coordinates have the same dimensions")

        self.structure = Structure(
            lattice=self.structure.lattice,
            species=self.structure.species,
            coords=new_coordinates,
            coords_are_cartesian=coords_are_cartesian,
        )

        self.vars.update(
            {
                "xcart": self.grab_cartesian_coordinates(),
                "xred": self.grab_reduced_coordinates(),
            }
        )

    def perturbations(
        self, perturbation: np.ndarray, coords_are_cartesian: bool = False
    ):
        """
        Apply a perturbation to the unit cell's coordinates and return a new AbinitUnitCell.

        This method adds the given perturbation to the current coordinates (either cartesian or reduced,
        as determined by the coords_are_cartesian flag) and returns a copy of the unit cell with the
        updated coordinates.

        Parameters:
            perturbation (np.ndarray):
                Array representing the perturbation to be applied. Its shape must match the shape of the
                current coordinate array.
            coords_are_cartesian (bool, optional):
                If True, the perturbation is applied to cartesian coordinates; otherwise, it is applied to reduced
                coordinates. Defaults to False.

        Returns:
            AbinitUnitCell: A new instance of AbinitUnitCell with the perturbed coordinates.

        Raises:
            ValueError: If the perturbation array shape does not match the shape of the current coordinates.
        """
        perturbation = np.array(perturbation, dtype=float)
        if perturbation.shape != self.grab_cartesian_coordinates().shape:
            raise ValueError(
                "Perturbation must have the same shape as the coordinates."
            )

        if coords_are_cartesian:
            new_coordinates = self.vars["xcart"] + perturbation
        else:
            new_coordinates = self.vars["xred"] + perturbation

        copy_cell = self.copy_abinit_unit_cell()
        copy_cell.change_coordinates(
            new_coordinates=new_coordinates, coords_are_cartesian=coords_are_cartesian
        )
        return copy_cell
