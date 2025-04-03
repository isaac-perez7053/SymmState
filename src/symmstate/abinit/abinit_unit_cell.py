from symmstate.unit_cell_module import UnitCell
import numpy as np
import copy
from pymatgen.core import Element
# from symmstate.utils.logger import configure_logging
from typing import Optional, List
from symmstate.utils.parsers import AbinitParser
from pymatgen.core import Structure, Lattice, Element

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
        
        self.vars = {}  # Initialize empty dict first

        # Validate initialization method
        init_methods = [abi_file, unit_cell, smodes_input]
        if sum(x is not None for x in init_methods) != 1:
            raise ValueError("Specify exactly one initialization method")

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
            super().__init__(
                acell=self.vars['acell'],
                rprim=self.vars['rprim'],
                coordinates=coordinates,
                coords_are_cartesian=coords_are_cartesian,
                elements=self._convert_znucl_typat()
            )

        elif unit_cell:
            if not isinstance(unit_cell, Structure):
                raise TypeError("unit_cell must be a Structure")
            super().__init__(structure=unit_cell)
            self._derive_abinit_parameters()
        else:
            super().__init__(
                smodes_file=smodes_input,
                target_irrep=target_irrep,
                symm_prec=symm_prec
            )

    def _store_abinit_parameters(self):
        """Store Abinit-specific parameters from parsed file"""
        self.znucl = self.vars.get('znucl')
        self.typat = self.vars.get('typat')
        self.ecut = self.vars.get('ecut')

    def _derive_abinit_parameters(self):
        """Derive parameters and populate vars"""
        self.rprim = self.structure.lattice.matrix
        self.natom = len(self.structure)
        self.znucl = sorted({e.Z for e in self.structure.species})
        self.typat = [self.znucl.index(s.Z) + 1 for s in self.structure.species]
        self.ntypat = len(self.znucl)
        
        # Update vars
        self.vars.update({
            "acell": list(self.structure.lattice.abc),
            "rprim": self.structure.lattice.matrix.tolist(),
            "znucl": self.znucl,
            "typat": self.typat
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

        # Store additional Abinit-specific parameters
        self.znucl = self.vars.get('znucl')
        self.typat = self.vars.get('typat')
        self.ecut = self.vars.get('ecut')

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
        self.structure = Structure(
            lattice=self.structure.lattice,
            species=self.structure.species,
            coords=new_coordinates,
            coords_are_cartesian=coords_are_cartesian
        )
        self.coordinates_xcart = self.structure.cart_coords
        self.coordinates_xred = self.structure.frac_coords

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
        if perturbation.shape != self.coordinates_xred.shape:
            raise ValueError(
                "Perturbation must have the same shape as the coordinates."
            )

        if coords_is_cartesian:
            new_coordinates = self.coordinates_xcart + perturbation
        else:
            new_coordinates = self.coordinates_xred + perturbation

        copy_cell = self.copy_abinit_unit_cell()
        # Calculate new coordinates by adding the perturbation
        copy_cell.change_coordinates(
            new_coordinates=new_coordinates, coords_are_cartesian=coords_is_cartesian
        )

        return copy_cell
    
    def __repr__(): 
        super().__repr__
