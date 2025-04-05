import unittest
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

# Import the Settings class so we can use its values.
from symmstate.config.settings import Settings
# Import your SmodesProcessor from its package.
from symmstate.flpz.smodes_processor import SmodesProcessor

# Dummy Slurm class to simulate job management.
class DummySlurm:
    def __init__(self):
        self.running_jobs = []
    def wait_for_jobs_to_finish(self, check_time=60, check_once=False):
        # For testing, simply do nothing.
        pass

class TestSmodesProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for file I/O.
        self.test_dir = tempfile.TemporaryDirectory()
        self.orig_dir = os.getcwd()
        os.chdir(self.test_dir.name)
        
        # Create a dummy Abinit file.
        self.abi_file = "dummy.abi"
        with open(self.abi_file, "w") as f:
            f.write(
                "acell 1.0 1.0 1.0\n"
                "rprim\n"
                "1.0 0.0 0.0\n"
                "0.0 1.0 0.0\n"
                "0.0 0.0 1.0\n"
                "xred\n"
                "0.0 0.0 0.0\n"
            )
        
        # Create a dummy SMODES input file.
        self.smodes_input = "dummy_smodes.in"
        with open(self.smodes_input, "w") as f:
            f.write("dummy smodes input")
        
        self.target_irrep = "A1"
        self.slurm_obj = DummySlurm()
        
        # Patch the symmetry‐adapted basis function to return dummy values.
        # The dummy tuple returns:
        # transmodes=True, isir=True, israman=True, type_count=[1],
        # type_list=["H"], num_sam=1, mass_list=[1.0],
        # pos_mat_cart = [[0.0, 0.0, 0.0]],
        # dist_mat = [[[1.0, 0.0, 0.0]]],
        # sam_atom_label = ["H"]
        dummy_smodes_output = (None, (True, True, True, [1], ["H"], 1, [1.0],
                                       np.array([[0.0, 0.0, 0.0]]),
                                       np.array([[[1.0, 0.0, 0.0]]]),
                                       ["H"]))
        self.symm_patch = patch(
            'symmstate.utils.symmetry_adapted_basis.SymmAdaptedBasis.symmatry_adapted_basis',
            return_value=dummy_smodes_output
        )
        self.mock_symm = self.symm_patch.start()
        
        # Create an instance of SmodesProcessor.
        self.processor = SmodesProcessor(
            abi_file=self.abi_file,
            smodes_input=self.smodes_input,
            target_irrep=self.target_irrep,
            slurm_obj=self.slurm_obj,
            disp_mag=0.001,
            symm_prec=1e-5,
            unstable_threshold=-20,
        )
        
        # Set required dummy attributes on the contained AbinitFile.
        self.processor.abinit_file.coordinates_xred = np.array([[0.0, 0.0, 0.0]])
        self.processor.abinit_file.coordinates_xcart = np.array([[0.0, 0.0, 0.0]])
        self.processor.abinit_file.natom = 1
        self.processor.abinit_file.ntypat = 1
        # Override methods to prevent real file operations.
        self.processor.abinit_file.write_custom_abifile = lambda fname, cont, coords_are_cartesian=False: None
        self.processor.abinit_file.run_abinit = lambda **kwargs: None
        self.processor.abinit_file.change_coordinates = lambda new_coords, coords_are_cartesian: None
        
        # Create dummy job output files for _perform_calculations.
        with open("dist_0.abo", "w") as f:
            f.write("Header\ncartesian forces (eV/Angstrom)\n0 0.1 0.1 0.1\nFooter\n")
        with open("dist_1.abo", "w") as f:
            f.write("Header\ncartesian forces (eV/Angstrom)\n0 0.2 0.2 0.2\nFooter\n")
        self.processor.jobs_ran_abo = ["dist_0.abo", "dist_1.abo"]

    def tearDown(self):
        self.symm_patch.stop()
        os.chdir(self.orig_dir)
        self.test_dir.cleanup()

    def test_run_smodes_success(self):
        # Patch subprocess.run in run_smodes to simulate a successful execution.
        with patch("subprocess.run") as mock_run:
            dummy_result = MagicMock(returncode=0, stdout="SMODES output", stderr="")
            mock_run.return_value = dummy_result
            output = self.processor.run_smodes(self.smodes_input)
            self.assertEqual(output, "SMODES output")

    def test_run_smodes_failure(self):
        # Simulate failure when running SMODES.
        with patch("subprocess.run") as mock_run:
            dummy_result = MagicMock(returncode=1, stdout="", stderr="error")
            mock_run.return_value = dummy_result
            with self.assertRaises(RuntimeError):
                self.processor.run_smodes(self.smodes_input)

    def test_imaginary_frequencies(self):
        # Set fc_evals so that one eigenvalue is below the unstable threshold.
        self.processor.fc_evals = np.array([-25, 10])
        unstable = self.processor._imaginary_frequencies()
        self.assertEqual(unstable, [0])

    def test_unstable_phonons(self):
        # Set fc_evals and dummy phonon vectors.
        self.processor.fc_evals = np.array([-25, 10])
        # Create dummy phonon vectors with shape (num_sam, natom, 3); here num_sam = 1, natom = 1.
        self.processor.phonon_vecs = np.array([[[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=np.float64)
        unstable = self.processor.unstable_phonons()
        self.assertIsInstance(unstable, list)
        self.assertEqual(len(unstable), 1)
        self.assertEqual(unstable[0].shape, (1, 3))

    def test_symmadapt(self):
        # Patch _loop_modes and _perform_calculations to bypass external actions.
        with patch.object(self.processor, "_loop_modes") as mock_loop, \
             patch.object(self.processor, "_perform_calculations") as mock_calc, \
             patch.object(self.processor, "unstable_phonons", return_value="dummy unstable"):
            result = self.processor.symmadapt()
            self.assertEqual(result, "dummy unstable")

    def test_perform_calculations(self):
        # Test _perform_calculations with dummy job files.
        self.processor.abinit_file.natom = 1
        self.processor.disp_mag = 0.001
        self.processor._perform_calculations()
        self.assertIsNotNone(self.processor.dyn_freqs)
        self.assertIsNotNone(self.processor.fc_evals)
        self.assertIsNotNone(self.processor.phonon_vecs)
        self.assertIsNotNone(self.processor.red_mass)

if __name__ == '__main__':
    unittest.main()

