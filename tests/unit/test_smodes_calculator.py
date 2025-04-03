import unittest
import os
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from symmstate.flpz.smodes_processor import SmodesProcessor

class TestSmodesProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for all file I/O during testing.
        self.test_dir = tempfile.TemporaryDirectory()
        self.orig_dir = os.getcwd()
        os.chdir(self.test_dir.name)
        
        # Create dummy ABI and SMODES input files.
        self.dummy_abi = "dummy.abi"
        with open(self.dummy_abi, "w") as f:
            f.write("acell 1.0 1.0 1.0\n"
                    "rprim\n"
                    "1.0 0.0 0.0\n"
                    "0.0 1.0 0.0\n"
                    "0.0 0.0 1.0\n"
                    "xred\n"
                    "0.0 0.0 0.0\n")
        
        self.dummy_smodes = "dummy_smodes.in"
        with open(self.dummy_smodes, "w") as f:
            f.write("dummy smodes input")
        
        # Patch the _symmetry_adapted_basis method to return controlled dummy values.
        # The dummy tuple returns:
        # transmodes=True, isir=True, israman=True, type_count=[1],
        # type_list=["H"], num_sam=1, mass_list=[1.0],
        # pos_mat_cart = [[0,0,0]], dist_mat = [[[1.0, 0, 0]]], sam_atom_label=["H"]
        dummy_smodes_output = (None, (True, True, True, [1], ["H"], 1, [1.0],
                                       np.array([[0.0, 0.0, 0.0]]),
                                       np.array([[[1.0, 0.0, 0.0]]]),
                                       ["H"]))
        self.symm_patch = patch('smodes_processor.AbinitFile._symmetry_adapted_basis',
                                  return_value=dummy_smodes_output)
        self.mock_symm = self.symm_patch.start()
        
        # For the SMODES executable check in run_smodes, we want a file to exist.
        # For simplicity, we make the dummy ABI file serve that role.
        self.smodes_exe = self.dummy_abi

        # Instantiate a SmodesProcessor with dummy parameters.
        self.processor = SmodesProcessor(
            abi_file=self.dummy_abi,
            smodes_input=self.dummy_smodes,
            target_irrep="A1",
            smodes_path=self.smodes_exe,  # dummy; exists
            host_spec="mpirun -np 1",
            disp_mag=0.001,
            symm_prec=1e-5,
            b_script_header_file="dummy header",
            unstable_threshold=-20,
        )
        
        # Set up required attributes on the contained AbinitFile.
        # (Coordinates and natom are needed for _loop_modes and calculations.)
        self.processor.abinit_file.coordinates_xred = np.array([[0.0, 0.0, 0.0]])
        self.processor.abinit_file.coordinates_xcart = np.array([[0.0, 0.0, 0.0]])
        self.processor.abinit_file.natom = 1

        # Override methods that would perform external actions.
        self.processor.abinit_file.write_custom_abifile = lambda fname, cont, coords_are_cartesian=False: None
        self.processor.abinit_file.run_abinit = lambda **kwargs: None
        self.processor.abinit_file.wait_for_jobs_to_finish = lambda t: None
        self.processor.abinit_file.change_coordinates = lambda new_coords, coords_are_cartesian: None

        # Create dummy output files for _perform_calculations.
        # They must contain a line "cartesian forces (eV/Angstrom)" followed by one force line per atom.
        with open("dist_0.abo", "w") as f:
            f.write("Header\ncartesian forces (eV/Angstrom)\n0 0.1 0.1 0.1\nFooter\n")
        with open("dist_1.abo", "w") as f:
            f.write("Header\ncartesian forces (eV/Angstrom)\n0 0.2 0.2 0.2\nFooter\n")
        self.processor.jobs_ran_abo = ["dist_0.abo", "dist_1.abo"]
    
    def tearDown(self):
        self.symm_patch.stop()
        os.chdir(self.orig_dir)
        self.test_dir.cleanup()

    def test_stabilize_matrix(self):
        # Create a matrix with a high condition number.
        A = np.array([[1e-10, 1.0],
                      [1.0, 1.0]], dtype=np.float32)
        stabilized = self.processor.stabilize_matrix(A.copy(), threshold=10)
        self.assertLess(np.linalg.cond(stabilized), np.linalg.cond(A))
        
    def test_run_smodes_success(self):
        # Patch subprocess.run in run_smodes to simulate a successful execution.
        with patch("subprocess.run") as mock_run:
            dummy_result = MagicMock()
            dummy_result.returncode = 0
            dummy_result.stdout = "SMODES output"
            mock_run.return_value = dummy_result
            output = self.processor.run_smodes(self.dummy_smodes)
            self.assertEqual(output, "SMODES output")
    
    def test_run_smodes_failure(self):
        # Simulate failure in running SMODES.
        with patch("subprocess.run") as mock_run:
            dummy_result = MagicMock()
            dummy_result.returncode = 1
            dummy_result.stderr = "error"
            mock_run.return_value = dummy_result
            with self.assertRaises(RuntimeError):
                self.processor.run_smodes(self.dummy_smodes)
    
    def test_imaginary_frequencies(self):
        # Set fc_evals with one eigenvalue below the unstable threshold.
        self.processor.fc_evals = np.array([-25, 10])
        unstable = self.processor._imaginary_frequencies()
        self.assertEqual(unstable, [0])
    
    def test_unstable_phonons(self):
        # Provide dummy phonon_vecs and fc_evals so that one mode is unstable.
        self.processor.fc_evals = np.array([-25, 10])
        # Create dummy phonon vectors: shape (num_sam, natom, 3); here num_sam = 1, natom = 1.
        self.processor.phonon_vecs = np.array([[[0.0, 1.0, 0.0]],
                                                [[1.0, 0.0, 0.0]]], dtype=np.float64)
        unstable = self.processor.unstable_phonons()
        self.assertIsInstance(unstable, list)
        self.assertEqual(len(unstable), 1)
        self.assertEqual(unstable[0].shape, (self.processor.abinit_file.natom, 3))
    
    def test_symmadapt(self):
        # Patch _loop_modes, _perform_calculations, and unstable_phonons.
        with patch.object(self.processor, "_loop_modes") as mock_loop, \
             patch.object(self.processor, "_perform_calculations") as mock_calc, \
             patch.object(self.processor, "unstable_phonons", return_value="dummy unstable"):
            result = self.processor.symmadapt()
            self.assertEqual(result, "dummy unstable")
    
    def test_perform_calculations(self):
        # Call _perform_calculations using our dummy output files.
        # With one atom, force difference from dist_0 to dist_1 is (0.2-0.1)=0.1.
        self.processor.abinit_file.natom = 1
        self.processor.disp_mag = 0.001
        # _perform_calculations will read our dummy files.
        self.processor._perform_calculations()
        # Check that key attributes are set.
        self.assertIsNotNone(self.processor.dyn_freqs)
        self.assertIsNotNone(self.processor.fc_evals)
        self.assertIsNotNone(self.processor.phonon_vecs)
        self.assertIsNotNone(self.processor.red_mass)

if __name__ == "__main__":
    unittest.main()
