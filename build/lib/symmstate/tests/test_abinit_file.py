import unittest
from unittest.mock import patch, mock_open, MagicMock
import tempfile
import os
from flpz.abinit import AbinitUnitCell
import numpy as np


class TestAbinitUnitCell(unittest.TestCase):

    def setUp(self):
        # Setup that runs before each test. You could create a temporary file here if needed.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.abi_file_path = os.path.join(self.temp_dir.name, "test.abi")

        with open(self.abi_file_path, "w") as f:
            f.write("ecut 30\n")
            f.write("ecutsm 0.1\n")
            f.write("nshiftk 2\n")
            f.write("shiftk 1.0 1.0 1.0\n")
            f.write("nband 40\n")
            f.write("nstep 50\n")
            f.write("diemac 4.0\n")
            f.write("toldfe 1d-6\n")
            f.write("ixc 11\n")
            f.write('pp_dirpath "/some/path/"\n')
            f.write('pseudos "pseudo1.pseudo"\n')
            f.write("kptrlatt\n0 0 0\n")

    def tearDown(self):
        # Tear down that runs after each test
        self.temp_dir.cleanup()

    @patch("symmstate.unit_cell_module.UnitCell", autospec=True)
    def test_init(self, MockUnitCell):
        # Patch the UnitCell base class and its constructor
        mock_parent_instance = MockUnitCell.return_value

        cell = AbinitUnitCell(abi_file=self.abi_file_path)

        # Check if parent class' constructor was called
        MockUnitCell.assert_called_with(
            abi_file=self.abi_file_path, smodes_file=None, target_irrep=None
        )

        # Assert the attributes are correctly initialized from the .abi file
        self.assertEqual(cell.ecut, 30)
        self.assertEqual(cell.ecutsm, 0.1)
        self.assertEqual(cell.nshiftk, 2)
        self.assertEqual(cell.shiftk, [1.0, 1.0, 1.0])
        self.assertEqual(cell.nband, 40)
        self.assertEqual(cell.nstep, 50)
        self.assertEqual(cell.diemac, 4.0)
        self.assertEqual(cell.toldfe, "1d-6")
        self.assertEqual(cell.ixc, 11)

    def test_process_atoms(self):
        elements = ["H", "O", "H"]
        num_atoms, ntypat, typat, znucl = AbinitUnitCell._process_atoms(elements)

        self.assertEqual(num_atoms, 3)
        self.assertEqual(ntypat, 2)  # H and O
        self.assertEqual(typat, [1, 2, 1])  # Indices corresponding to H=1, O=2
        self.assertEqual(znucl, [1, 8])  # Atomic numbers for Hydrogen and Oxygen

    def test_copy_abinit_unit_cell(self):
        original_cell = AbinitUnitCell(abi_file=self.abi_file_path)
        copied_cell = original_cell.copy_abinit_unit_cell()

        self.assertNotEqual(id(original_cell), id(copied_cell))
        self.assertEqual(original_cell.ecut, copied_cell.ecut)

    @patch("subprocess.run")
    def test_all_jobs_finished(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        cell = AbinitUnitCell(abi_file=self.abi_file_path)
        cell.runningJobs = [12345]

        all_finished = cell.all_jobs_finished()

        self.assertTrue(all_finished)
        mock_subprocess_run.assert_called_once_with(
            ["squeue", "-j", str(12345)], capture_output=True, text=True
        )

    def test_change_coordinates(self):
        original_cell = AbinitUnitCell(abi_file=self.abi_file_path)
        fake_structure = MagicMock()
        fake_structure.cart_coords = np.array([[0, 0, 0]])
        fake_structure.frac_coords = np.array([[0.5, 0.5, 0.5]])
        original_cell.structure = fake_structure

        new_coordinates = np.array([[0.1, 0.1, 0.1]])
        original_cell.change_coordinates(new_coordinates, coords_are_cartesian=True)

        np.testing.assert_array_equal(original_cell.coordinates_xcart, new_coordinates)
        np.testing.assert_array_equal(
            original_cell.coordinates_xred, fake_structure.frac_coords
        )


if __name__ == "__main__":
    unittest.main()
