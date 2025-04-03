import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pymatgen.core import Structure, Lattice, Element
from symmstate.abinit.abinit_file import AbinitFile


# Dummy variables to be returned by the AbinitParser
dummy_vars = {
    "acell": [1.0, 1.0, 1.0],
    "rprim": [[1.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]],
    "xred": [[0.0, 0.0, 0.0]],
    "znucl": [1],
    "typat": [1]
}

class TestAbinitFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test file I/O.
        self.test_dir = tempfile.mkdtemp()
        self.orig_dir = os.getcwd()
        os.chdir(self.test_dir)

        # Create a dummy ABI file with minimal valid contents.
        self.dummy_abi_file = os.path.join(self.test_dir, "dummy.abi")
        with open(self.dummy_abi_file, "w") as f:
            f.write("acell 1.0 1.0 1.0\n")
            f.write("rprim\n")
            f.write("  1.0 0.0 0.0\n")
            f.write("  0.0 1.0 0.0\n")
            f.write("  0.0 0.0 1.0\n")
            f.write("xred\n")
            f.write("  0.0 0.0 0.0\n")
            f.write("znucl 1\n")
            f.write("typat 1\n")
        
        # Patch the AbinitParser.parse_abinit_file so that the dummy file returns dummy_vars.
        patcher = patch("symmstate.utils.parsers.AbinitParser.parse_abinit_file",
                        return_value=dummy_vars)
        self.addCleanup(patcher.stop)
        self.mock_parser = patcher.start()

        # Supply a dummy header to avoid errors in SlurmFile.
        self.abinit = AbinitFile(abi_file=self.dummy_abi_file, batch_script_header_file="dummy header")

        # Patch the missing write_batch_script method to return a dummy value.
        self.abinit.write_batch_script = lambda input_file, batch_name, host_spec, log: True

        # Set additional attributes used in file writing and execution.
        self.abinit.ecut = 20.0
        self.abinit.ecutsm = None
        self.abinit.nshiftk = 1
        self.abinit.kptrlatt = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.abinit.shiftk = [0.0, 0.0, 0.0]
        self.abinit.nband = 10
        self.abinit.nstep = 50
        self.abinit.diemac = 0.0
        self.abinit.ixc = 7
        self.abinit.conv_criteria = ["tolvrs", "1.0d-8"]
        # Simulate pseudopotential availability.
        self.abinit.pseudopotentials = ["H.psp8"]
        # Set a dummy SLURM batch header.
        self.abinit.batch_header = "dummy header"
    
    def tearDown(self):
        # Return to the original working directory and remove the temporary directory.
        os.chdir(self.orig_dir)
        shutil.rmtree(self.test_dir)

    def test_get_unique_filename(self):
        filename = "dummy.txt"
        with open(filename, "w") as f:
            f.write("dummy")
        unique_name = AbinitFile._get_unique_filename(filename)
        self.assertNotEqual(unique_name, filename)
        with open(unique_name, "w") as f:
            f.write("dummy")
        unique_name2 = AbinitFile._get_unique_filename(filename)
        self.assertNotEqual(unique_name2, filename)
        self.assertNotEqual(unique_name2, unique_name)

    def test_write_custom_abifile(self):
        header = "!! This is a test header\n"
        output_file = "test_input"
        self.abinit.write_custom_abifile(output_file, header, coords_are_cartesian=False)
        expected_file = f"{output_file}.abi"
        self.assertTrue(os.path.exists(expected_file))
        with open(expected_file, "r") as f:
            content = f.read()
        self.assertIn("acell", content)
        self.assertIn("rprim", content)
        self.assertIn("xred", content)
        self.assertIn("natom", content)
        self.assertIn("pp_dirpath", content)
        self.assertIn("pseudos", content)

    @patch("subprocess.run")
    def test_run_abinit(self, mock_run):
        dummy_result = MagicMock()
        dummy_result.returncode = 0
        dummy_result.stdout = "Submitted batch job 12345"
        mock_run.return_value = dummy_result

        input_file = "dummy_abinit"
        self.abinit.batch_header = "dummy header"
        self.abinit.run_abinit(
            input_file=input_file,
            batch_name="dummy_job",
            batch_script_header_file="dummy header",
            host_spec="mpirun -np 4",
            delete_batch_script=True,
            log="dummy.log",
        )
        self.assertTrue(mock_run.called)
        self.assertIn(12345, self.abinit.running_jobs)

    @patch("subprocess.run")
    def test_run_piezo_calculation(self, mock_run):
        dummy_result = MagicMock()
        dummy_result.returncode = 0
        dummy_result.stdout = "Submitted batch job 54321"
        mock_run.return_value = dummy_result

        self.abinit.batch_header = "dummy header"
        self.abinit.run_piezo_calculation(host_spec="mpirun -np 4")
        self.assertIn(54321, self.abinit.running_jobs)

    @patch("subprocess.run")
    def test_run_flexo_calculation(self, mock_run):
        dummy_result = MagicMock()
        dummy_result.returncode = 0
        dummy_result.stdout = "Submitted batch job 67890"
        mock_run.return_value = dummy_result

        self.abinit.batch_header = "dummy header"
        self.abinit.run_flexo_calculation(host_spec="mpirun -np 4")
        self.assertIn(67890, self.abinit.running_jobs)

    @patch("subprocess.run")
    def test_run_energy_calculation(self, mock_run):
        dummy_result = MagicMock()
        dummy_result.returncode = 0
        dummy_result.stdout = "Submitted batch job 98765"
        mock_run.return_value = dummy_result

        self.abinit.batch_header = "dummy header"
        self.abinit.run_energy_calculation(host_spec="mpirun -np 4")
        self.assertIn(98765, self.abinit.running_jobs)

    def test_grab_energy(self):
        energy_file = "dummy_energy.abo"
        energy_value = -123.456E+00
        content = f"Some text\n total_energy: {energy_value}\n some other text"
        with open(energy_file, "w") as f:
            f.write(content)
        self.abinit.grab_energy(energy_file)
        self.assertAlmostEqual(self.abinit.energy, energy_value)

    def test_parse_tensor(self):
        tensor_str = "1.0 2.0 3.0\n4.0 5.0 6.0\n"
        result = self.abinit.parse_tensor(tensor_str)
        expected = np.array([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_grab_flexo_tensor(self):
        flexo_file = "dummy_flexo.abo"
        tensor_block = "\n".join(["xx 1.0 2.0 3.0 4.0 5.0 6.0"] * 9)
        content = f"Header\nTOTAL flexoelectric tensor (units= nC/m)\n  xx yy zz yz xz xy\n{tensor_block}\nFooter"
        with open(flexo_file, "w") as f:
            f.write(content)
        self.abinit.grab_flexo_tensor(flexo_file)
        self.assertEqual(self.abinit.flexo_tensor.shape, (9, 6))

    def test_grab_piezo_tensor(self):
        piezo_file = "dummy_piezo.abo"
        clamped_block = "1.0 0.0\n0.0 1.0\n"
        relaxed_block = "0.5 0.0\n0.0 0.5\n"
        content = (f"Proper piezoelectric constants (clamped ion) (unit:c/m^2)\n{clamped_block}\n"
                   f"Proper piezoelectric constants (relaxed ion) (unit:c/m^2)\n{relaxed_block}\n")
        with open(piezo_file, "w") as f:
            f.write(content)
        self.abinit.grab_piezo_tensor(piezo_file)
        np.testing.assert_array_almost_equal(
            self.abinit.piezo_tensor_clamped,
            np.array([[1.0, 0.0], [0.0, 1.0]])
        )
        np.testing.assert_array_almost_equal(
            self.abinit.piezo_tensor_relaxed,
            np.array([[0.5, 0.0], [0.0, 0.5]])
        )

    def test_copy_abinit_file(self):
        copy_instance = self.abinit.copy_abinit_file()
        self.assertIsNot(copy_instance, self.abinit)
        self.assertEqual(copy_instance.natom, self.abinit.natom)
        self.assertEqual(copy_instance.ntypat, self.abinit.ntypat)
        self.assertEqual(copy_instance.znucl, self.abinit.znucl)
        self.assertEqual(copy_instance.typat, self.abinit.typat)

if __name__ == "__main__":
    unittest.main()

