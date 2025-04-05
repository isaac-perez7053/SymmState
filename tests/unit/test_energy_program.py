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
      - Directly input acell (array), rprim (ndarray), coordinates (ndarray), etc.
      - Use a symmetry adapted basis (if smodes_file and target_irrep are provided)
      - Use pymatgen structure

    Public Methods:
      - find_space_group(): Returns space group of the UnitCell
      - grab_reduced_coordinates(): Returns the reduced coordinates of the UnitCell
      - grab_cartesian_coordinates(): Returns the cartesian coordinates of the UnitCell
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
        Initialize the class through either:
        1. Direct structural parameters, or
        2. SMODES file and target irreducible representation

        If a SMODES file is provided, it will override any manual input of the parameters.
        """
        # If SMODES input is provided, ignore manual inputs and load from SMODES.
        if smodes_file and target_irrep:
            if not os.path.isfile(smodes_file):
                raise FileNotFoundError(f"SMODES file not found: {smodes_file}")
            # Load parameters from SMODES file and ignore any manual input.
            params, _ = SymmAdaptedBasis.symmatry_adapted_basis(
                smodes_file, target_irrep, symm_prec
            )
            acell, rprim, coordinates, coords_are_cartesian, elements = params

        if structure:
            self.structure = structure
        else:
            # Validate that all required manual parameters are provided.
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
            
            # Build structure from manual parameters.
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

        # Always set coordinates
        self.coordinates_xred = self.structure.frac_coords
        self.coordinates_xcart = self.structure.cart_coords
        self.clean_reduced_coordinates()

    def grab_reduced_coordinates(self):
        """Grabs the reduced coordinates of the UnitCell"""
        return np.array(self.structure.frac_coords)

    def grab_cartesian_coordinates(self):
        """Grabs the cartesian coordinates of the UnitCell"""
        return np.array(self.structure.cart_coords)

    def find_space_group(self):
        """Calculates and returns the space group of the unit cell."""
        analyzer = SpacegroupAnalyzer(self.structure)
        return (analyzer.get_space_group_number(), analyzer.get_space_group_symbol())

    def perturbations(self, perturbation, coords_are_cartesian=False):
        """
        Apply a given perturbation to the unit cell coordinates and return a new UnitCell.

        Args:
            perturbation (np.ndarray): A numpy array representing the perturbation to be applied.
            coords_are_cartesian (bool): If True, treats perturbation as cartesian, else reduced.

        Returns:
            UnitCell: A new instance of UnitCell with perturbed coordinates.
        """

        # Ensure the perturbation has the correct shape
        perturbation = np.array(perturbation, dtype=float)
        if perturbation.shape != self.structure.frac_coords.shape:
            raise ValueError(
                "Perturbation must have the same shape as the fractional coordinates."
            )

        # Calculate new fractional coordinates by adding the perturbation
        if coords_are_cartesian:
            new_frac_coords = self.structure.cart_coords + perturbation
        else:
            new_frac_coords = self.structure.frac_coords + perturbation

        # Create a new Structure object with the updated fractional coordinates
        perturbed_structure = Structure(
            lattice=self.structure.lattice,
            species=self.structure.species,
            coords=new_frac_coords,
            coords_are_cartesian=coords_are_cartesian,  # Ensure coordinates are treated as fractional
        )

        # Return a new instance of UnitCell with the perturbed structure
        return UnitCell(structure=perturbed_structure)

    def _round_to_nearest(self, value):
        # Convert value to Decimal for better precision
        d = Decimal(str(value))
        # Round to the nearest number with up to a reasonable precision
        rounded_decimal = d.quantize(Decimal('1e-15'), rounding=ROUND_HALF_UP)
        # Convert the Decimal to a float
        return float(rounded_decimal)

    def clean_reduced_coordinates(self):
        # Copy the array to avoid modifying the original
        cleaned_arr = np.copy(self.structure.frac_coords)
        
        # Function to round, and check tiny numbers
        def clean_value(x):
            # Round up if ending with .9999... to the nearest integer
            if abs(x - self._round_to_nearest(x)) < 1e-9:
                return self._round_to_nearest(x)
            # Set values close to zero (on the order of e-17) to zero
            elif abs(x) < 1e-16:
                return 0.0
            # Return the number itself if no conditions are met
            else:
                return x
        
        # Vectorize the cleaning function for the numpy array
        vectorized_clean_value = np.vectorize(clean_value)

        # Apply function to each element
        cleaned_arr = vectorized_clean_value(cleaned_arr)

        # Update structure with cleaned coordinates.
        self.structure = Structure(
            lattice=self.structure.lattice,
            species=self.structure.species,
            coords=cleaned_arr,
            coords_are_cartesian=False,
        )
    
    def __repr__(self):
        return str(self.structure)
