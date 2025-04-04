import unittest
import tempfile
import os
import numpy as np
from unittest.mock import patch, MagicMock

from symmstate.config.settings import settings
from symmstate.slurm_file import SlurmFile
from symmstate.flpz.energy.energy_program import EnergyProgram

# Dummy output from AbinitParser.parse_abinit_file that satisfies convergence criterion.
dummy_abinit_dict = {
    "acell": [1.0, 1.0, 1.0],
    "rprim": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
    "xred": [[0, 0, 0]],
    "znucl": [1],
    "typat": [1],
    "toldfe": 1e-6,
    "nband": 10,
    "natom": 1,
    "ntypat": 1,
    "ixc": 7,
    "conv_critera": "toldfe"
}

# Dummy SlurmFile instance
dummy_slurm = SlurmFile(sbatch_header_source="#SBATCH --time=24:00:00\n#SBATCH --nodes=1", num_processors=1)
dummy_slurm.running_jobs = []  # ensure running_jobs attribute exists

# Dummy SmodesProcessor that returns a non-empty list to simulate unstable phonons.
class DummySmodesProcessor:
    def __init__(self, *args, **kwargs):
        self.abinit_file = MagicMock()
        # Dummy arrays to simulate phonon displacement vectors, force constant evals, and dynamical frequencies.
        self.phonon_vecs = np.array([[[0.1, 0.2, 0.3]]])
        self.fc_evals = np.array([1.0])
        self.dyn_freqs = [[0.5, 30]]
    def symmadapt(self):
        # Return a list with one dummy unstable phonon displacement array.
        return [np.array([[0.1, 0.1, 0.1]])]

# Dummy Perturbations class that simulates the perturbation calculations.
class DummyPerturbations:
    def __init__(self, *args, **kwargs):
        # For testing, we set some dummy values.
        self.list_amps = [0.0, 0.25, 0.5]
        self.results = {
            "energies": [100, 150, 200],
            "flexo": [np.array([[1]]), np.array([[2]]), np.array([[3]])],
            "piezo": {"clamped": [np.array([[1.5]]), np.array([[2.5]]), np.array([[3.5]])],
                      "relaxed": [np.array([[2]]), np.array([[3]]), np.array([[4]])]}
        }
    def generate_perturbations(self):
        # In a dummy version, do nothing.
        return
    def calculate_energy_of_perturbations(self):
        return
    def calculate_piezo_of_perturbations(self):
        return
    def calculate_flexo_of_perturbations(self):
        return
    def data_analysis(self, save_plot, filename):
        # Instead of plotting, simply log a message.
        pass

# Test class for EnergyProgram.
class TestEnergyProgram(unittest.TestCase):
    def setUp(self):
        # Create temporary dummy files for the ABI file and SMODES input.
        self.test_dir = tempfile.TemporaryDirectory()
        self.abi_file = os.path.join(self.test_dir.name, "dummy.abi")
        with open(self.abi_file, "w") as f:
            f.write("Dummy content for abi file")
        self.smodes_input = os.path.join(self.test_dir.name, "dummy_smodes.txt")
        with open(self.smodes_input, "w") as f:
            f.write("Dummy content for smodes input")
        self.target_irrep = "irrepsample"
        self.name = "TestEnergyProgram"
        self.num_datapoints = 3
        self.min_amp = 0.0
        self.max_amp = 0.5
        self.symm_prec = settings.SYMM_PREC
        self.disp_mag = 0.001
        self.unstable_threshold = -20

        # Use the dummy slurm_obj.
        self.slurm_obj = dummy_slurm

        # Patch AbinitParser.parse_abinit_file to return our dummy dictionary.
        patcher = patch("symmstate.utils.parsers.AbinitParser.parse_abinit_file", return_value=dummy_abinit_dict)
        self.addCleanup(patcher.stop)
        self.mock_parse = patcher.start()

        # Patch SmodesProcessor with our dummy version.
        patcher2 = patch("symmstate.flpz.smodes_processor.SmodesProcessor", new=DummySmodesProcessor)
        self.addCleanup(patcher2.stop)
        patcher2.start()

        # Patch Perturbations with our dummy version.
        patcher3 = patch("symmstate.flpz.perturbations.Perturbations", new=DummyPerturbations)
        self.addCleanup(patcher3.stop)
        patcher3.start()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_run_program_with_unstable_phonons(self):
        # Create an instance of EnergyProgram with our dummy parameters and slurm_obj.
        energy_prog = EnergyProgram(
            name=self.name,
            num_datapoints=self.num_datapoints,
            abi_file=self.abi_file,
            min_amp=self.min_amp,
            max_amp=self.max_amp,
            smodes_input=self.smodes_input,
            target_irrep=self.target_irrep,
            slurm_obj=self.slurm_obj,
            symm_prec=self.symm_prec,
            disp_mag=self.disp_mag,
            unstable_threshold=self.unstable_threshold,
        )
        # Run the program.
        energy_prog.run_program()

        # Check that the smodes processor was set.
        self.assertIsNotNone(energy_prog.get_smodes_processor())
        # Since our DummySmodesProcessor.symmadapt returns one unstable phonon,
        # we expect that a Perturbations instance was created.
        perturbations = energy_prog.get_perturbations()
        self.assertIsInstance(perturbations, list)
        self.assertGreater(len(perturbations), 0)

    def test_run_program_no_unstable_phonons(self):
        # Patch DummySmodesProcessor.symmadapt to return an empty list.
        with patch.object(DummySmodesProcessor, "symmadapt", return_value=[]):
            energy_prog = EnergyProgram(
                name=self.name,
                num_datapoints=self.num_datapoints,
                abi_file=self.abi_file,
                min_amp=self.min_amp,
                max_amp=self.max_amp,
                smodes_input=self.smodes_input,
                target_irrep=self.target_irrep,
                slurm_obj=self.slurm_obj,
                symm_prec=self.symm_prec,
                disp_mag=self.disp_mag,
                unstable_threshold=self.unstable_threshold,
            )
            energy_prog.run_program()
            # With no unstable phonons, the perturbations list should remain empty.
            self.assertEqual(len(energy_prog.get_perturbations()), 0)

    def test_get_smodes_processor(self):
        energy_prog = EnergyProgram(
            name=self.name,
            num_datapoints=self.num_datapoints,
            abi_file=self.abi_file,
            min_amp=self.min_amp,
            max_amp=self.max_amp,
            smodes_input=self.smodes_input,
            target_irrep=self.target_irrep,
            slurm_obj=self.slurm_obj,
            symm_prec=self.symm_prec,
            disp_mag=self.disp_mag,
            unstable_threshold=self.unstable_threshold,
        )
        energy_prog.run_program()
        smodes_proc = energy_prog.get_smodes_processor()
        self.assertIsNotNone(smodes_proc)

    def test_get_perturbations(self):
        energy_prog = EnergyProgram(
            name=self.name,
            num_datapoints=self.num_datapoints,
            abi_file=self.abi_file,
            min_amp=self.min_amp,
            max_amp=self.max_amp,
            smodes_input=self.smodes_input,
            target_irrep=self.target_irrep,
            slurm_obj=self.slurm_obj,
            symm_prec=self.symm_prec,
            disp_mag=self.disp_mag,
            unstable_threshold=self.unstable_threshold,
        )
        energy_prog.run_program()
        perturbations = energy_prog.get_perturbations()
        self.assertIsInstance(perturbations, list)

if __name__ == "__main__":
    unittest.main()
