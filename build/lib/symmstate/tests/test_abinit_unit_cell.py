import unittest
import tempfile
import numpy as np
from symmstate.abinit import (
    AbinitUnitCell,
)  # Replace `module_name` with the correct module name
import os


class TestAbinitUnitCellInitialization(unittest.TestCase):

    def create_mock_abifile(self):
        """
        Creates a temporary Abinit file with known content for testing.
        Returns the path to the temporary file.
        """
        abifile_content = """\
acell 10.0 10.0 10.0
rprim
  1.0 0.0 0.0
  0.0 1.0 0.0
  0.0 0.0 1.0
natom 1
typat 1
ntypat 1
znucl 6
xred
  0.0 0.0 0.0
"""
        tmp_file = tempfile.NamedTemporaryFile(delete=False, mode="w+t")
        tmp_file.write(abifile_content)
        tmp_file.close()
        return tmp_file.name

    def test_initialize_from_abifile(self):
        # Create a mock abifile
        abifile_path = "tests/example1_file.abi"

        try:
            # Initialize a new AbinitUnitCell using the mock file
            unit_cell = AbinitUnitCell(abi_file=abifile_path)

            # Expected values based on the mock file
            # expected_num_atoms = 5
            # expected_atom_types = np.array([1], dtype=int)
            # expected_znucl = np.array([6], dtype=int)
            # expected_typat = np.array([1], dtype=int)

            # Assertions to confirm the state of the UnitCell
            # np.testing.assert_array_almost_equal(unit_cell.acell, expected_acell)
            # np.testing.assert_array_almost_equal(unit_cell.rprim, expected_rprim)
            # np.testing.assert_array_almost_equal(unit_cell.coordinates, expected_coordinates)
            # self.assertEqual(unit_cell.num_atoms, expected_num_atoms)
            # np.testing.assert_array_equal(unit_cell.atom_types, expected_atom_types)
            # np.testing.assert_array_equal(unit_cell.znucl, expected_znucl)
            # np.testing.assert_array_equal(unit_cell.typat, expected_typat)
            header_file_content = """
    kptopt 1
"""
            unit_cell.write_custom_abifile(
                output_file="example_abifile.abi", header_file=header_file_content
            )
        #           unit_cell.write_batch_script(batch_script_header_file="tests/b-script-preamble.txt", output_file="output.sh", host="mpirun -np 4 abinit <")

        finally:
            # Clean up the temporary file
            os.remove(abifile_path)


if __name__ == "__main__":
    unittest.main()
