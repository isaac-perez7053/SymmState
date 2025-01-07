import numpy as np
import unittest
from pathlib import Path
import sys

from shared.abinit_file import AbinitFile  # Assuming this is where your class is defined

# To run this file, run python3 -m unittest shared.test_abinit_file from the main module

class TestAbinitFile(unittest.TestCase):

    def test_abinit_file_1(self):
        self.maxDiff = None
        # Define the path dynamically as in the main function
        example1_path = (Path(__file__).resolve().parent / "example1_file.abi").resolve()

        # Initialize AbinitFile object
        abinit_file1 = AbinitFile(filepath=str(example1_path))
        
        # Check rprim vectors explicitely
        expected_rprim_1 = "[[0.       7.254681 0.      ]\n [0.       0.       7.254681]\n [7.254681 0.       0.      ]]"
        
        self.assertEqual(str(abinit_file1.unit_cell.rprim), expected_rprim_1)

        # Expected output for unit cell
        expected_unit_cell_1 = "UnitCell(acell=[1. 1. 1.], coord_type='reduced', rprim=[[0.       7.254681 0.      ]\n [0.       0.       7.254681]\n [7.254681 0.       0.      ]], coordinates=[[0.  0.  0. ]\n [0.5 0.5 0.5]\n [0.5 0.5 0. ]\n [0.5 0.  0.5]\n [0.  0.5 0.5]], num_atoms=5, atom_types=3, znucl=[20 22  8], typat=[1 2 3 3 3], header_path='/Users/isaacperez/Downloads/Personal_Projects/abinit-9.10.3/flpz/tests/example1_file.abi.header')"

        # Expected space group for the cell
        expected_space_group_1 = "Pm-3m (221)"

        # Expected converted coordinates for the cell
        expected_cartesian_coordinates_1 = "[[0.        0.        0.       ]\n [3.6273405 3.6273405 3.6273405]\n [0.        3.6273405 3.6273405]\n [3.6273405 3.6273405 0.       ]\n [3.6273405 0.        3.6273405]]"
        expected_coord_type_1 = "cartesian"

        # Check if the unit cell matches the expected output
        self.assertEqual(str(abinit_file1.unit_cell), str(expected_unit_cell_1))

        # Check if the space group is calculated correctly.
        self.assertEqual(str(abinit_file1.unit_cell.findSpaceGroup()), expected_space_group_1)

        # Check if the coordinate conversion is properly working
        abinit_file1_oldcoords = abinit_file1.unit_cell.coordinates
        abinit_file1.unit_cell.convertToXcart()
        self.assertEqual(str(abinit_file1.unit_cell.coordinates), expected_cartesian_coordinates_1)
        self.assertEqual(str(abinit_file1.unit_cell.coord_type), expected_coord_type_1)

        # Check if the space group is calculated correctly.
        self.assertEqual(str(abinit_file1.unit_cell.findSpaceGroup()), expected_space_group_1)

        abinit_file1.unit_cell.convertToXred()
        # Check if the conversion is reversible
        self.assertEqual(str(abinit_file1.unit_cell.coordinates), str(abinit_file1_oldcoords))

        # Check if the space group is calculated correctly.
        self.assertEqual(str(abinit_file1.unit_cell.findSpaceGroup()), expected_space_group_1)

    def test_abinit_file_2(self):
        self.maxDiff = None
        # Define the path dynamically as in the main function
        example2_path = (Path(__file__).resolve().parent / "example2_file.abi").resolve()

        # Initialize AbinitFile object
        abinit_file2 = AbinitFile(filepath=str(example2_path))
        
        # Check rprim vectors explicitely
        expected_rprim_2 = "[[ 7.254681 -7.254681  0.      ]\n [ 7.254681  7.254681  0.      ]\n [ 0.        0.        7.254681]]"
        
        self.assertEqual(str(abinit_file2.unit_cell.rprim), expected_rprim_2)


        # Expected space group for the cell
        expected_space_group_2 = "Pm-3m (221)"

        # Expected output for unit cell
        expected_unit_cell_2 = (
        "UnitCell(acell=[1. 1. 1.], coord_type='cartesian', rprim=[[ 7.254681 -7.254681  0.      ]\n [ 7.254681  7.254681  0.      ]\n [ 0.        0.        7.254681]], coordinates=[[ 0.         0.         0.       ]\n"
        " [ 7.254681   0.         0.       ]\n [ 3.6273405  3.6273405  3.6273405]\n"
        " [ 3.6273405 -3.6273405  3.6273405]\n [ 7.254681  -3.6273405  3.6273405]\n"
        " [ 7.254681   3.6273405  3.6273405]\n [ 3.6273405  3.6273405  0.       ]\n"
        " [ 3.6273405 -3.6273405  0.       ]\n [ 3.6273405  0.         3.6273405]\n"
        " [10.8820215  0.         3.6273405]], num_atoms=10, atom_types=3, znucl=[20 22  8], "
        "typat=[1 1 2 2 3 3 3 3 3 3], header_path='/Users/isaacperez/Downloads/Personal_Projects/abinit-9.10.3/flpz/tests/example2_file.abi.header')"
    )
        # Expected converted coordinates for the cell
        expected_cartesian_coordinates_2 =  "[[0.   0.   0.  ]\n [0.5  0.5  0.  ]\n [0.   0.5  0.5 ]\n [0.5  0.   0.5 ]\n [0.75 0.25 0.5 ]\n [0.25 0.75 0.5 ]\n [0.   0.5  0.  ]\n [0.5  0.   0.  ]\n [0.25 0.25 0.5 ]\n [0.75 0.75 0.5 ]]"
        expected_coord_type_2 = "reduced"

        # Check if the unit cell matches the expected output
        self.assertEqual(str(abinit_file2.unit_cell), expected_unit_cell_2)

        # Check if the space group is calculated correctly 
        self.assertEqual(str(abinit_file2.unit_cell.findSpaceGroup()),str( expected_space_group_2))

        # Check if the reduced coordinates are claculated correctly
        abinit_file2_oldcoords = abinit_file2.unit_cell.coordinates
        abinit_file2.unit_cell.convertToXred()
        self.assertEqual(str(abinit_file2.unit_cell.coordinates), expected_cartesian_coordinates_2)
        self.assertEqual(str(abinit_file2.unit_cell.coord_type), expected_coord_type_2)

        # Check if the space group is calculated correctly 
        self.assertEqual(str(abinit_file2.unit_cell.findSpaceGroup()),str( expected_space_group_2))


        abinit_file2.unit_cell.convertToXcart()
        # Check if the conversion is reversible
        self.assertEqual(str(abinit_file2.unit_cell.coordinates), str(abinit_file2_oldcoords))
        
        # Check if the space group is calculated correctly 
        self.assertEqual(str(abinit_file2.unit_cell.findSpaceGroup()),str( expected_space_group_2))

    def test_error_handling(self):
        # Simulate an error in case the file does not exist
        with self.assertRaises(Exception):
            invalid_path = (Path(__file__).resolve().parent / "../Testing/invalid_file.abi").resolve()
            abinit_file_invalid = AbinitFile(filepath=str(invalid_path))

if __name__ == "__main__":
    unittest.main()
