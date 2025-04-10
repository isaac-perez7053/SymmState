from symmstate.unit_cell_module import UnitCell
import numpy as np
import copy
from pymatgen.core import Element
from typing import Optional, List
from symmstate.utils.abinit_parser import AbinitParser
from pymatgen.core import Structure, Lattice, Element
from symmstate.utils.misc import Misc

class AbinitUnitCell(UnitCell):
    """
    Extends UnitCell with Abinit-specific functionality and initialization paths.
    """
    def __init__(
        self,
        abi_file: Optional[str] = None,
        unit_cell: Optional[Structure] = None,
        *,
        smodes_input: Optional[str] = None,
        target_irrep: Optional[str] = None,
        symm_prec: float = 1e-5
    ):
        self.vars = {}  # Initialize empty dictionary first


        if abi_file:
            self.abi_file = abi_file
            self.vars = AbinitParser.parse_abinit_file(abi_file)
            
            # Handle coordinates properly
            if 'xred' in self.vars:
                coordinates = self.vars['xred']
                coords_are_cartesian = False
            elif 'xcart' in self.vars:
                coordinates = self.vars['xcart']
                coords_are_cartesian = True
            else:
                raise ValueError("No coordinates found in Abinit file")

            # Initialize parent class with parsed parameters
            if (smodes_input is not None) and (target_irrep is not None):
                super().__init__(
                    smodes_file=smodes_input,
                    target_irrep=target_irrep,
                    symm_prec=symm_prec
                )
                # Update parameters with new cell
                self.update_unit_cell_parameters()
 
            else:
                super().__init__(
                    acell=self.vars['acell'],
                    rprim=self.vars['rprim'],
                    coordinates=coordinates,
                    coords_are_cartesian=coords_are_cartesian,
                    elements=self._convert_znucl_typat(),
                )

        elif unit_cell:
            if not isinstance(unit_cell, Structure):
                raise TypeError("unit_cell must be a Structure")
            super().__init__(structure=unit_cell)
            self._derive_abinit_parameters()
        else:
            raise ValueError("Provide a valid input for initailization")

        # Ensure an instance of xred and xcart are included in the variable dictionary 
        if 'xred' not in self.vars:
            self.vars['xred'] = np.array(self.structure.frac_coords)
        elif 'xcart' not in self.vars:
            self.vars['xcart'] = np.array(self.structure.cart_coords)

    def update_unit_cell_parameters(self):
        """
        Updates unit cell parameters in self.vars based on the current structure,
        preserving the original ordering of atomic sites.
        
        This method calculates:
        - natom: total number of atoms,
        - znucl: a sorted list of unique atomic numbers,
        - typat: re-computed from the original ordering of species (but now using the sorted order),
        - ntypat: number of unique species,
        - nband: computed number of bands.
        
        It also reorders the pseudos in self.vars["pseudos"] so that each pseudopotential
        corresponds correctly with the new sorted order of atomic numbers.
        
        The procedure is as follows:
        1. Determine the original unique atomic numbers (in order of first appearance)
            and the corresponding pseudopotential order.
        2. Sort the unique atomic numbers.
        3. For each sorted atomic number, look up its index in the original order and 
            reassemble the pseudos list accordingly.
        4. Recompute the 'typat' list so that each atom’s type index refers to the index
            in the sorted list.
        """
        # Get the original list of sites and extract species.
        sites = self.structure.sites
        species = [site.specie for site in sites]

        natom = len(sites)
        # Preserve the original unique order as they appear.
        original_znucl = list(dict.fromkeys([s.Z for s in species]))  # e.g. [20, 8, 22]
        # Original typat follows the original order.
        original_typat = [original_znucl.index(s.Z) + 1 for s in species]
        ntypat = len(original_znucl)

        # Calculate the number of bands using an external routine.
        nband = Misc.calculate_nband(self.structure)

        # Get the original pseudos list (assumed to be set in self.vars).
        original_pseudos = self.vars.get("pseudos", [])
        if len(original_pseudos) != ntypat:
            raise ValueError("The number of pseudopotentials does not match the number of unique atom types.")

        # Now sort the unique atomic numbers.
        sorted_znucl = sorted(original_znucl)  # e.g. [8, 20, 22]

        # Reassemble the pseudos list so that for each sorted atomic number, we
        # pick the pseudopotential corresponding to the position in the original list.
        new_pseudos = [original_pseudos[original_znucl.index(z)] for z in sorted_znucl]

        # Recompute typat using the new sorted order.
        new_typat = [sorted_znucl.index(s.Z) + 1 for s in species]

        # Finally, update self.vars with the new parameters.
        self.vars.update({
            "natom": natom,
            "znucl": sorted_znucl,
            "typat": new_typat,
            "ntypat": ntypat,
            "nband": nband,
            "pseudos": new_pseudos,
        })


    def _derive_abinit_parameters(self):
        """Derive parameters and populate vars"""
        rprim = self.structure.lattice.matrix
        natom = len(self.structure)
        znucl = sorted({e.Z for e in self.structure.species})
        typat = [znucl.index(s.Z) + 1 for s in self.structure.species]
        ntypat = len(znucl)
        
        # Update vars
        self.vars.update({
            "acell": self.structure.lattice.abc,
            "rprim": np.array(rprim),
            "znucl": znucl,
            "typat": typat,
            "natom": natom,
            "ntypat": ntypat
        })

    def _init_from_abinit_vars(self):
        """Initialize from parsed Abinit variables"""
        # Extract critical structural parameters
        acell = self.vars['acell']
        rprim = self.vars['rprim']
        
        # Handle coordinate system
        if 'xcart' in self.vars:
            coordinates = self.vars['xcart']
            coords_are_cartesian = True
        elif 'xred' in self.vars:
            coordinates = self.vars['xred']
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
            elements=elements
        )


    def _convert_znucl_typat(self) -> List[str]:
        """Convert znucl/typat to element symbols"""
        if 'znucl' not in self.vars or 'typat' not in self.vars:
            raise ValueError("Missing znucl or typat in Abinit file")
            
        znucl = self.vars['znucl']
        typat = self.vars['typat']
        return [Element.from_Z(znucl[t-1]).symbol for t in typat]

    @property
    def abinit_parameters(self) -> dict:
        """Return copy of vars if available"""
        return self.vars.copy() if hasattr(self, "vars") else {}
        # # -----------------------------
        # # Abinit Specific Calculations
        # # -----------------------------

        # # Energy of the unit cell
        # self.energy =  None

        # # Electric properties of the cell
        # self.piezo_tensor_clamped = None
        # self.piezo_tensor_relaxed = None
        # self.flexo_tensor = None

    # --------------------------
    # Initialization Methods
    # --------------------------

    @staticmethod
    def _process_atoms(atom_list):
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
        Creates a deep copy of the current AbinitUnitCell instance.

        Returns:
            AbinitUnitCell: A new instance that is a deep copy of the current instance.
        """
        # Perform a deep copy to ensure all nested objects are also copied
        copied_cell = copy.deepcopy(self)
        return copied_cell

    def change_coordinates(self, new_coordinates, coords_are_cartesian=False):
        """Update coordinates directly without super() call"""

        if not isinstance(new_coordinates, np.ndarray):
            raise TypeError("Ensure that the new coordinates are a numpy array")
        
        elif new_coordinates.shape != self.vars['xred'].shape:
            raise ValueError("Ensure that the coordinates have the same dimensions")

        self.structure = Structure(
            lattice=self.structure.lattice,
            species=self.structure.species,
            coords=new_coordinates,
            coords_are_cartesian=coords_are_cartesian
        )

        # Update variable dictionary
        if coords_are_cartesian:
            self.vars['xcart'] = new_coordinates
        else:
            self.vars['xred'] = new_coordinates

    def perturbations(self, perturbation, coords_is_cartesian=False):
        """
        Applies a given perturbation to the unit cell's coordinates and returns a new AbinitUnitCell object.

        Args:
            pert (np.ndarray): Array representing the perturbation to be applied to current coordinates.

        Returns:
            AbinitUnitCell: A new instance of UnitCell with updated (perturbed) coordinates.
        """

        # Ensure the perturbation has the correct shape
        perturbation = np.array(perturbation, dtype=float)
        if perturbation.shape != self.grab_cartesian_coordinates().shape:
            raise ValueError(
                "Perturbation must have the same shape as the coordinates."
            )

        if coords_is_cartesian:
            new_coordinates = self.vars['xcart'] + perturbation
        else:
            new_coordinates = self.vars['xred'] + perturbation

        copy_cell = self.copy_abinit_unit_cell()
        # Calculate new coordinates by adding the perturbation
        copy_cell.change_coordinates(
            new_coordinates=new_coordinates, coords_are_cartesian=coords_is_cartesian
        )

        return copy_cell
    
    def __repr__(): 
        super().__repr__
