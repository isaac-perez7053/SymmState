import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch
from pymatgen.core import Structure, Lattice
from symmstate import UnitCell # Update import path as needed

class TestUnitCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a basic unit cell for tests that don't modify state
        cls.basic_cell = UnitCell(
            acell=[2.0, 2.0, 2.0],
            rprim=np.eye(3),
            coordinates=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
            coords_are_cartesian=False,
            elements=["Si", "Si"]
        )

    def test_direct_initialization(self):
        """Test direct initialization with structural parameters"""
        self.assertEqual(self.basic_cell.structure.num_sites, 2)
        self.assertTrue(np.allclose(
            self.basic_cell.structure.lattice.matrix,
            np.eye(3) * 2
        ))

    def test_missing_parameters(self):
        """Test missing required structural parameters"""
        with self.assertRaises(ValueError) as context:
            UnitCell(
                acell=[2.0, 2.0, 2.0],
                rprim=np.eye(3),
                coordinates=None,  # Missing required
                coords_are_cartesian=False,
                elements=["Si", "Si"]
            )

    # @patch('symmstate.SymmAdaptedBasis.symmatry_adapted_basis')
    # def test_smodes_initialization(self, mock_basis):
    #     """Test initialization from SMODES file"""
    #     mock_data = (
    #         [2.0, 2.0, 2.0],
    #         np.eye(3),
    #         np.array([[0,0,0], [0.5,0.5,0.5]]),
    #         False,
    #         ["Si", "Si"]
    #     )
    #     mock_basis.return_value = (mock_data, None)

    #     with tempfile.NamedTemporaryFile() as tmp:
    #         uc = UnitCell(
    #             smodes_file=tmp.name,
    #             target_irrep="GM4-",
    #             symm_prec=1e-6
    #         )
            
    #         self.assertEqual(uc.structure.num_sites, 2)
    #         mock_basis.assert_called_once_with(tmp.name, "GM1", 1e-6)

    # def test_smodes_file_not_found(self):
    #     """Test missing SMODES file handling"""
    #     with self.assertRaises(FileNotFoundError):
    #         UnitCell(
    #             smodes_file="nonexistent.smodes",
    #             target_irrep="Γ1"
    #         )

    def test_conflicting_initialization(self):
        """Test mixed initialization parameters"""
        with tempfile.NamedTemporaryFile() as tmp:
            with self.assertRaises(ValueError) as context:
                UnitCell(
                    acell=[2.0, 2.0, 2.0],  # Should conflict
                    smodes_file=tmp.name,
                    target_irrep="GM4-"
                )
            self.assertIn("Structural parameters cannot be provided",
                         str(context.exception))

    def test_space_group(self):
        """Test space group identification"""
        sg_number, sg_symbol = self.basic_cell.find_space_group()
        self.assertEqual(sg_number, 229)  
        self.assertIn("Im-3m", sg_symbol)  

    def test_coordinate_transformations(self):
        """Test coordinate system conversions"""
        # Test reduced coordinates
        frac_coords = self.basic_cell.grab_reduced_coordinates()
        self.assertTrue(np.allclose(frac_coords, [[0,0,0], [0.5,0.5,0.5]]))
        
        # Test cartesian coordinates
        cart_coords = self.basic_cell.grab_cartesian_coordinates()
        expected = np.array([[0,0,0], [1.0,1.0,1.0]])
        self.assertTrue(np.allclose(cart_coords, expected))

    def test_perturbations(self):
        """Test coordinate perturbation functionality"""
        perturbation = np.array([[0.1, 0, 0], [0, 0, 0]])
        new_uc = self.basic_cell.perturbations(perturbation)
        
        new_frac = new_uc.grab_reduced_coordinates()
        expected = np.array([[0.1, 0, 0], [0.5,0.5,0.5]])
        self.assertTrue(np.allclose(new_frac, expected))
        
        # Verify original structure remains unchanged
        original_frac = self.basic_cell.grab_reduced_coordinates()
        self.assertTrue(np.allclose(original_frac, [[0,0,0], [0.5,0.5,0.5]]))

    def test_invalid_perturbation_shape(self):
        """Test invalid perturbation shape handling"""
        with self.assertRaises(ValueError) as context:
            bad_perturbation = np.array([0.1, 0, 0])  # Wrong shape
            self.basic_cell.perturbations(bad_perturbation)
        self.assertIn("must have the same shape", str(context.exception))

    def test_coordinate_cleaning(self):
        """Test cleaning of fractional coordinates"""
        dirty_coords = np.array([
            [0.9999999999, 1.0000000001, -0.0000000001],
            [1e-17, -1e-18, 0.5]
        ])
        
        uc = UnitCell(
            acell=[2.0, 2.0, 2.0],
            rprim=np.eye(3),
            coordinates=dirty_coords,
            coords_are_cartesian=False,
            elements=["Si", "Si"]
        )
        
        cleaned = uc.grab_reduced_coordinates()
        expected = np.array([
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.5]
        ])
        self.assertTrue(np.allclose(cleaned, expected))

    def test_rounding_edge_cases(self):
        """Test edge cases in coordinate rounding"""
        uc = UnitCell(
            acell=[2.0, 2.0, 2.0],
            rprim=np.eye(3),
            coordinates=np.array([[0.4999999999, 0.5000000001, 1e-16]]),
            coords_are_cartesian=False,
            elements=["Si"]
        )
        
        cleaned = uc.grab_reduced_coordinates()
        expected = np.array([[0.5, 0.5, 0.0]])
        self.assertTrue(np.allclose(cleaned, expected))

    def test_repr(self):
        """Test string representation"""
        rep = self.basic_cell.__repr__()
        self.assertIn("Full Formula (Si2)", rep)
        self.assertIn("Sites (2)", rep)

if __name__ == '__main__':
    unittest.main()