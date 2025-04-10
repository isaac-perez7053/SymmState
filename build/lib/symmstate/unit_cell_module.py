import numpy as np
import os
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from decimal import Decimal, ROUND_HALF_UP
from symmstate import SymmStateCore
from typing import Optional, List
from symmstate.utils.symmetry_adapted_basis import SymmAdaptedBasis


class UnitCell(SymmStateCore):
    """
    Defines the UnitCell class which contains all the necessary information of a UnitCell.

    Initialization:
      - You can provide the acell, rprim, coordinates, etc. manually (or via a pymatgen Structure),
        or specify a SMODES file and target irreducible representation. When the SMODES inputs are
        provided, the symmetry-adapted basis is calculated and manual inputs are ignored.
      
    Public Methods:
      - find_space_group(): Returns space group of the UnitCell.
      - grab_reduced_coordinates(): Returns the reduced coordinates of the UnitCell.
      - grab_cartesian_coordinates(): Returns the cartesian coordinates of the UnitCell.
    """

    def __init__(
        self,
        structure: Optional[Structure] = None,
        acell: Optional[List[float]] = None,
        rprim: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None,
        coords_are_cartesian: Optional[bool] = None,
        elements: Optional[List[str]] = None,
        *,
        smodes_file: Optional[str] = None,
        target_irrep: Optional[str] = None,
        symm_prec: float = 1e-5
    ):
        """
        Initialize the UnitCell.

        If a SMODES file and target irreducible representation are provided, the symmetry‐adapted basis
        is calculated and its output is used (any manual inputs are ignored). Otherwise, if a pre‐built
        structure is provided, it is used directly; if not, all manual parameters must be provided.
        """
        # If SMODES input is provided, always use it to determine the structural parameters.
        if smodes_file and target_irrep:
            if not os.path.isfile(smodes_file):
                raise FileNotFoundError(f"SMODES file not found: {smodes_file}")
            # Overwrite any manual parameters with those returned by SMODES.
            params, _ = SymmAdaptedBasis.symmatry_adapted_basis(smodes_file, target_irrep, symm_prec)
            acell, rprim, coordinates, coords_are_cartesian, elements = params

        if structure:
            self.structure = structure
        else:
            # Ensure that all required parameters are provided when no structure is given.
            required_fields = {
                "acell": acell,
                "rprim": rprim,
                "coordinates": coordinates,
                "coords_are_cartesian": coords_are_cartesian,
                "elements": elements
            }
            missing = [k for k, v in required_fields.items() if v is None]
            if missing:
                raise ValueError(f"Missing parameters: {', '.join(missing)}")
            
            # Build structure from provided parameters.
            acell = np.array(acell, dtype=float)
            rprim = np.array(rprim, dtype=float)
            coordinates = np.array(coordinates, dtype=float)
            elements = np.array(elements, dtype=str)

            lattice = Lattice(rprim * acell)
            self.structure = Structure(
                lattice=lattice,
                species=elements,
                coords=coordinates,
                coords_are_cartesian=coords_are_cartesian
            )

        # Set coordinates from the structure.
        self.coordinates_xred = self.structure.frac_coords
        self.coordinates_xcart = self.structure.cart_coords
        self.clean_reduced_coordinates()

    def grab_reduced_coordinates(self):
        """Return a copy of the reduced (fractional) coordinates."""
        return np.array(self.structure.frac_coords)

    def grab_cartesian_coordinates(self):
        """Return a copy of the cartesian coordinates."""
        return np.array(self.structure.cart_coords)

    def find_space_group(self):
        """Calculate and return the space group of the unit cell."""
        analyzer = SpacegroupAnalyzer(self.structure)
        return (analyzer.get_space_group_number(), analyzer.get_space_group_symbol())

    def perturbations(self, perturbation, coords_are_cartesian=False):
        """
        Apply a given perturbation to the unit cell coordinates and return a new UnitCell.

        Args:
            perturbation (np.ndarray): Array representing the perturbation.
            coords_are_cartesian (bool): If True, perturbation is cartesian; otherwise, it is fractional.

        Returns:
            UnitCell: A new UnitCell instance with the perturbed coordinates.
        """
        perturbation = np.array(perturbation, dtype=float)
        if perturbation.shape != self.structure.frac_coords.shape:
            raise ValueError("Perturbation must match the shape of the fractional coordinates.")

        new_frac_coords = (self.structure.cart_coords + perturbation) if coords_are_cartesian else (self.structure.frac_coords + perturbation)
        perturbed_structure = Structure(
            lattice=self.structure.lattice,
            species=self.structure.species,
            coords=new_frac_coords,
            coords_are_cartesian=coords_are_cartesian
        )
        return UnitCell(structure=perturbed_structure)

    def _round_to_nearest(self, value):
        d = Decimal(str(value))
        rounded_decimal = d.quantize(Decimal('1e-15'), rounding=ROUND_HALF_UP)
        return float(rounded_decimal)

    def clean_reduced_coordinates(self):
        """Clean the fractional coordinates to remove tiny numerical noise."""
        cleaned_arr = np.copy(self.structure.frac_coords)

        def clean_value(x):
            if abs(x - self._round_to_nearest(x)) < 1e-9:
                return self._round_to_nearest(x)
            elif abs(x) < 1e-16:
                return 0.0
            else:
                return x

        vectorized_clean_value = np.vectorize(clean_value)
        cleaned_arr = vectorized_clean_value(cleaned_arr)

        self.structure = Structure(
            lattice=self.structure.lattice,
            species=self.structure.species,
            coords=cleaned_arr,
            coords_are_cartesian=False
        )

    def __repr__(self):
        return str(self.structure)

