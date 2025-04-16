import unittest
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the settings class so we can use its values.
from symmstate.config.symm_state_settings import settings

# Import your SmodesProcessor from its package.
from symmstate.flpz.smodes_processor import SmodesProcessor


# Dummy Slurm class to simulate job management.
class DummySlurm:
    def __init__(self):
        self.running_jobs = []

    def wait_for_jobs_to_finish(self, check_time=60, check_once=False):
        # For testing purposes, do nothing.
        pass


# A helper dummy function to simulate SMODES output.
def simulate_smodes_output():
    # We supply valid dummy parameters:
    dummy_params = [
        [3.0, 3.0, 3.0],  # acell
        np.array(
            [
                [0.0, 7.2546934054, 0.0],
                [0.0, 0.0, 7.2546934054],
                [7.2546934054, 0.0, 0.0],
            ]
        ),  # rprim
        np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0]]
        ),  # coordinates (3 atoms)
        False,  # coords_are_cartesian
        ["Ca", "Ti", "O"],  # elements
    ]
    # The second element is a tuple of dummy additional data.
    dummy_extra = (
        True,
        True,
        True,
        [1],
        ["H"],
        1,
        [1.0],
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[[1.0, 0.0, 0.0]]]),
        ["H"],
    )
    return (dummy_params, dummy_extra)


class TestSmodesProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for file I/O.
        self.test_dir = tempfile.TemporaryDirectory()
        self.orig_dir = os.getcwd()
        os.chdir(self.test_dir.name)

        # Create a dummy Abinit file.
        content = """
useylm 1  # Use of spherical harmonics
kptopt 2  # Takes into account time-reversal symmetry.

# Allows the use of cells that are non-primitive
chkprim 0

#Definition of unit cell
#***********************
acell 3*1.0
xred
   0.0000000000 0.0000000000 0.0000000000
   0.5000000000 0.5000000000 0.5000000000
   0.5000000000 0.5000000000 0.0000000000
   0.5000000000 0.0000000000 0.5000000000
   0.0000000000 0.5000000000 0.5000000000
rprim
   0.0000000000 7.2546934054 0.0000000000
   0.0000000000 0.0000000000 7.2546934054
   7.2546934054 0.0000000000 0.0000000000

#Definition of atoms
#************************
natom 5
ntypat 3
znucl 20 22 8
typat 1 2 3*3

#Definition of the planewave basis set
#*************************************
ecut 70
ecutsm 0.5 #Smoothing energy needed for lattice parameter optimization.

#Definition of the k-point grid
#******************************
nshiftk 1
kptrlatt
 6   0   0
 0   6   0
 0   0   6
shiftk 0.5 0.5 0.5
nband 52

#Definition of SCF Procedure
#***************************
nstep 50
diemac 4.0
ixc -116133    #GGA specified by pseudopotential files
toldfe 1.0d-9

pp_dirpath "../"
pseudos "CaRev.psp8, TiRev.psp8, ORev.psp8"
"""
        self.abi_file = "dummy.abi"
        with open(self.abi_file, "w") as f:
            f.write(content)

        # Create a dummy SMODES input file.
        smodes_content = """
7.2546810033  7.2546810033  7.2546810033
221
1 1 1  90 90 90
3
Ca a
Ti b
O c
1
GM
"""
        self.smodes_input = "dummy_smodes.in"
        with open(self.smodes_input, "w") as f:
            f.write(smodes_content)

        self.target_irrep = "GM4-"
        self.slurm_obj = DummySlurm()

        # Patch the symmetry-adapted basis function to return our dummy SMODES output.
        self.symm_patch = patch(
            "symmstate.utils.symmetry_adapted_basis.SymmAdaptedBasis.symmatry_adapted_basis",
            return_value=simulate_smodes_output(),
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
        self.processor.abinit_file.vars["natom"] = 1
        self.processor.abinit_file.vars["ntypat"] = 1

        # Override methods to prevent real file operations.
        self.processor.abinit_file.write_custom_abifile = (
            lambda fname, cont, coords_are_cartesian=False, pseudos=[]: None
        )
        self.processor.abinit_file.run_abinit = lambda **kwargs: None
        self.processor.abinit_file.change_coordinates = (
            lambda new_coords, coords_are_cartesian: None
        )

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
        self.processor.freq_cm = [-200, 100]
        unstable = self.processor._imaginary_frequencies()
        self.assertEqual(unstable, [0])

    def test_unstable_phonons(self):
        # Set fc_evals and dummy phonon vectors.
        self.processor.freq_cm = [-25, 10]
        # Create dummy phonon vectors with shape (num_sam, natom, 3); here num_sam = 1, natom = 1.
        self.processor.phonon_vecs = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float64)
        unstable = self.processor.unstable_phonons()
        self.assertIsInstance(unstable, list)
        self.assertEqual(len(unstable), 1)
        self.assertEqual(unstable[0].shape, (1, 3))

    def test_symmadapt(self):
        # Patch _loop_modes and _perform_calculations to bypass external actions.
        with patch.object(self.processor, "_loop_modes") as mock_loop, patch.object(
            self.processor, "_perform_calculations"
        ) as mock_calc, patch.object(
            self.processor, "unstable_phonons", return_value="dummy unstable"
        ):
            result = self.processor.symmadapt()
            self.assertEqual(result, "dummy unstable")

    def test_perform_calculations(self):
        # Test _perform_calculations with dummy job files.
        self.processor.abinit_file.vars["natom"] = 1
        self.processor.disp_mag = 0.001
        self.processor._perform_calculations()
        self.assertIsNotNone(self.processor.dyn_freqs)
        self.assertIsNotNone(self.processor.fc_evals)
        self.assertIsNotNone(self.processor.phonon_vecs)
        self.assertIsNotNone(self.processor.red_mass)


if __name__ == "__main__":
    unittest.main()
