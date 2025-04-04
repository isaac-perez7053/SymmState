import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Import the program to test.
from symmstate.flpz.energy.electrotensor_program import ElectroTensorProgram

# Dummy classes to simulate behavior.
class DummySmodesProcessor:
    def __init__(self, *args, **kwargs):
        # Create a dummy abinit_file attribute with minimal required attributes.
        self.abinit_file = MagicMock()
        self.abinit_file.file_name = "dummy_smodes.abi"
        self.abinit_file.coordinates_xred = np.array([[0.0, 0.0, 0.0]])
        self.abinit_file.coordinates_xcart = np.array([[0.0, 0.0, 0.0]])
        # Set dummy SMODES outputs:
        self.phonon_vecs = np.array([[0.1, 0.2, 0.3]])  # One unstable phonon vector
        self.fc_evals = [1.0, 2.0, 3.0]
        self.dyn_freqs = [[1.0, 2.0]]
    
    def symmadapt(self):
        # Return a list with one dummy unstable phonon vector.
        return [np.array([0.1, 0.2, 0.3])]

class DummyPerturbations:
    def __init__(self, *args, **kwargs):
        # Set dummy attributes for testing.
        self.list_amps = [0.0, 0.5]
        self.results = {"energies": [10, 20], "piezo": {"clamped": [[[1]]], "relaxed": [[[2]]]} }
        self.list_flexo_tensors = [[ [0.1]*6 ] * 9]
        self.list_piezo_tensors_clamped = [[ [0.2]*6 ] * 9]
        self.list_piezo_tensors_relaxed = [[ [0.3]*6 ] * 9]
    
    def generate_perturbations(self):
        # For testing, simply do nothing.
        pass

    def calculate_energy_of_perturbations(self):
        pass

    def calculate_flexo_of_perturbations(self):
        pass

    def calculate_piezo_of_perturbations(self):
        pass

    def data_analysis(self, **kwargs):
        # Dummy implementation.
        return

    def record_data(self, filename):
        # Write a dummy file.
        with open(filename, "w") as f:
            f.write("dummy data")

class DummySlurm:
    def __init__(self):
        self.running_jobs = []

# Test cases for ElectroTensorProgram.
class TestElectroTensorProgram(unittest.TestCase):
    def setUp(self):
        # Create a dummy slurm object.
        self.dummy_slurm = DummySlurm()
        # Create an instance of ElectroTensorProgram with dummy parameters.
        self.energy_prog = ElectroTensorProgram(
            name="TestElectro",
            num_datapoints=2,
            abi_file="dummy.abi",
            min_amp=0.0,
            max_amp=0.5,
            smodes_input="dummy_smodes.in",
            target_irrep="A1",
            slurm_obj=self.dummy_slurm,
            symm_prec=1e-5,
            disp_mag=0.001,
            unstable_threshold=-20,
            piezo_calculation=False
        )
        # Patch update_abinit_file to avoid file operations during testing.
        self.energy_prog.update_abinit_file = lambda x: None

    @patch('symmstate.flpz.energy.electrotensor_program.SmodesProcessor')
    @patch('symmstate.flpz.energy.electrotensor_program.Perturbations')
    def test_run_program_with_unstable_phonons(self, MockPerturbations, MockSmodesProcessor):
        # Set up DummySmodesProcessor to return one unstable phonon.
        dummy_smodes = DummySmodesProcessor()
        MockSmodesProcessor.return_value = dummy_smodes
        # Configure DummyPerturbations to be used for each unstable phonon.
        MockPerturbations.side_effect = lambda *args, **kwargs: DummyPerturbations()

        # Run the program.
        self.energy_prog.run_program()

        # Check that the smodes processor was set.
        self.assertIsNotNone(self.energy_prog.get_smodes_processor())
        # Check that perturbations were created (should be one since one unstable phonon).
        perturbations = self.energy_prog.get_perturbations()
        self.assertEqual(len(perturbations), 1)

    @patch('symmstate.flpz.energy.electrotensor_program.SmodesProcessor')
    def test_run_program_no_unstable_phonons(self, MockSmodesProcessor):
        # Set up DummySmodesProcessor to simulate no unstable phonons.
        dummy_smodes = DummySmodesProcessor()
        dummy_smodes.symmadapt = lambda: []  # Return empty list.
        MockSmodesProcessor.return_value = dummy_smodes

        self.energy_prog.run_program()

        # Expect that no perturbations were created.
        self.assertEqual(len(self.energy_prog.get_perturbations()), 0)

    def test_get_smodes_processor(self):
        # Manually set a dummy smodes processor.
        dummy_smodes = DummySmodesProcessor()
        self.energy_prog._ElectroTensorProgram__smodes_processor = dummy_smodes
        self.assertEqual(self.energy_prog.get_smodes_processor(), dummy_smodes)

    def test_get_perturbations(self):
        # Manually set dummy perturbations.
        dummy_perturbations = [DummyPerturbations(), DummyPerturbations()]
        self.energy_prog._ElectroTensorProgram__perturbations = dummy_perturbations
        self.assertEqual(self.energy_prog.get_perturbations(), dummy_perturbations)

if __name__ == '__main__':
    unittest.main()

