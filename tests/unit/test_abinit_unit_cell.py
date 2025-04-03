import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch
from pymatgen.core import Structure, Lattice
from symmstate.abinit.abinit_unit_cell import AbinitUnitCell

class TestAbinitUnitCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a basic structure for testing
        cls.si_structure = Structure(
            lattice=Lattice.cubic(5.0),
            species=["Si", "Si"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
        )

        # Create a temporary Abinit input file
        cls.temp_abinit_file = tempfile.NamedTemporaryFile(delete=False)
        cls._create_dummy_abinit_file(cls.temp_abinit_file.name)
        cls.temp_abinit_file.close()

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.temp_abinit_file.name)

    @staticmethod
    def _create_dummy_abinit_file(path):
        content = """acell 5.0 5.0 5.0
rprim
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
xred
0.0 0.0 0.0
0.5 0.5 0.5
znucl 14
typat 1 1
ecut 20
"""
        with open(path, 'w') as f:
            f.write(content)

    def test_structure_initialization(self):
        """Test initialization with pymatgen Structure"""
        cell = AbinitUnitCell(unit_cell=self.si_structure)
        
        # Verify structure parameters
        self.assertEqual(len(cell.structure), 2)
        self.assertAlmostEqual(cell.structure.lattice.a, 5.0)
        
        # Verify derived Abinit parameters
        self.assertEqual(cell.natom, 2)
        self.assertEqual(cell.ntypat, 1)
        self.assertEqual(cell.znucl, [14])
        self.assertEqual(cell.typat, [1, 1])

    def test_abinit_file_initialization(self):
        """Test initialization from Abinit input file"""
        cell = AbinitUnitCell(abi_file=self.temp_abinit_file.name)
        
        # Verify critical parameters
        self.assertIn('rprim', cell.vars)
        self.assertIn('acell', cell.vars)
        self.assertTrue(np.allclose(cell.vars['rprim'], np.eye(3)))

    # @patch('symmstate.utils.symmetry_adapted_basis.SymmAdaptedBasis.symmatry_adapted_basis')
    # def test_symmetry_initialization(self, mock_basis):
    #     """Test symmetry-adapted basis initialization"""
    #     mock_params = (
    #         [5.0, 5.0, 5.0],  # acell
    #         np.eye(3),         # rprim
    #         np.array([[0,0,0], [0.5,0.5,0.5]]),  # coordinates
    #         False,              # coords_are_cartesian
    #         ["Si", "Si"]        # elements
    #     )
    #     mock_basis.return_value = (mock_params, None)
        
    #     cell = AbinitUnitCell(smodes_input="dummy.smodes", target_irrep="Γ1")
    #     self.assertEqual(len(cell.structure), 2)
    #     self.assertEqual(cell.structure.species[0].symbol, "Si")

    def test_conflicting_initialization(self):
        """Test multiple initialization sources"""
        with self.assertRaises(ValueError):
            AbinitUnitCell(
                abi_file=self.temp_abinit_file.name,
                unit_cell=self.si_structure
            )

    def test_invalid_structure_type(self):
        """Test invalid type for unit_cell parameter"""
        with self.assertRaises(TypeError):
            AbinitUnitCell(unit_cell={"invalid": "type"})

    def test_copy_method(self):
        """Test deep copy functionality"""
        original = AbinitUnitCell(unit_cell=self.si_structure)
        copy_cell = original.copy_abinit_unit_cell()
        
        # Verify they're different objects
        self.assertIsNot(original, copy_cell)
        # Verify structure equality
        self.assertTrue(original.structure.matches(copy_cell.structure))

    def test_coordinate_perturbation(self):
        """Test coordinate perturbation method"""
        original = AbinitUnitCell(unit_cell=self.si_structure)
        perturbation = np.array([[0.1, 0, 0], [0, 0, 0]])
        
        # Test fractional coordinates perturbation
        perturbed = original.perturbations(perturbation)
        new_coords = perturbed.grab_reduced_coordinates()
        expected = np.array([[0.1, 0, 0], [0.5, 0.5, 0.5]])
        self.assertTrue(np.allclose(new_coords, expected))

    def test_property_access(self):
        """Test access to derived properties"""
        cell = AbinitUnitCell(unit_cell=self.si_structure)
        self.assertIsInstance(cell.abinit_parameters, dict)
        self.assertIn('rprim', cell.abinit_parameters)

    def test_coordinate_conversion(self):
        """Test coordinate system handling"""
        cell = AbinitUnitCell(abi_file=self.temp_abinit_file.name)
        xred = cell.grab_reduced_coordinates()
        xcart = cell.grab_cartesian_coordinates()
        
        # Verify coordinate conversion
        self.assertTrue(np.allclose(xcart, xred @ cell.structure.lattice.matrix))

    def test_invalid_perturbation_shape(self):
        """Test invalid perturbation shape handling"""
        cell = AbinitUnitCell(unit_cell=self.si_structure)
        with self.assertRaises(ValueError):
            bad_perturbation = np.array([0.1, 0, 0])  # Wrong shape
            cell.perturbations(bad_perturbation)

if __name__ == '__main__':
    unittest.main()

# TODOOOOOO USE LOGGER: 

# Assuming the `configure_logging` function is already defined

# # Configure the logger
# logger = configure_logging(name="myapp", level=logging.DEBUG, file_path="app.log")

# # Now you can log messages at different levels
# logger.debug("This is a debug message")
# logger.info("This is an informational message")
# logger.warning("This is a warning message")
# logger.error("This is an error message")
# logger.critical("This is a critical error message")





# # FOR MISSING ECUT

# def test_missing_ecut_error():
#     with tempfile.NamedTemporaryFile(mode="w") as f:
#         f.write("nshiftk 3\n")  # Missing ecut
#         abi_file = f.name
    
#     with pytest.raises(Exception, match="ecut is missing"):
#         AbinitUnitCell(abi_file=abi_file)._initialize_convergence_from_file()


#     # FOR CREATING MOCK FILES

#     import tempfile

# def test_initialization_from_abi_file():
#     # Create a temporary Abinit input file
#     with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
#         f.write("ecut 20\nnshiftk 3\nixc 7\npp_dirpath 'pseudo'\npseudos 'Si.psp8 O.psp8'\n")
#         abi_file = f.name

#     # Initialize AbinitUnitCell
#     cell = AbinitUnitCell(abi_file=abi_file)
    
#     # Check parsed values
#     assert cell.ecut == 20
#     assert cell.nshiftk == 3
#     assert cell.ixc == 7
#     assert "Si.psp8" in cell.pseudopotentials
    # assert "O.psp8" in cell.pseudopotentials



#### ALLOW USER TO CUSTOM INITIALIZE THE UNIT CELL: 

# # Initialize with a user-defined unit cell
# custom_unit_cell = {
#     "a": 10,
#     "b": 20,
#     "c": 30,
#     "angles": (90, 90, 90)
# }



# TO RUN FILE

# pytest -v test_abinit_unit_cell.py