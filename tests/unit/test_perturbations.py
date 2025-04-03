import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the Perturbations class.
from symmstate.flpz.perturbations import Perturbations

# Dummy perturbed object class to simulate AbinitFile perturbation results.
class DummyPerturbation:
    def __init__(self, base_value):
        self.file_name = "dummy"
        self.running_jobs = [123]
        self.energy = None
        self.piezo_tensor_clamped = None
        self.piezo_tensor_relaxed = None
        self.flexo_tensor = None
        self.base_value = base_value

    def run_energy_calculation(self, host_spec):
        self.energy = 100 + self.base_value

    def run_piezo_calculation(self, host_spec):
        self.energy = 110 + self.base_value
        self.piezo_tensor_clamped = np.array([[1 + self.base_value]])
        self.piezo_tensor_relaxed = np.array([[2 + self.base_value]])

    def run_flexo_calculation(self, host_spec):
        self.energy = 120 + self.base_value
        self.flexo_tensor = np.array([[3 + self.base_value]])

    def run_mrgddb_file(self, content):
        pass

    def run_anaddb_file(self, input_str, piezo=False, flexo=False, peizo=False):
        if flexo:
            return "dummy_flexo_output"
        elif peizo:
            return "dummy_piezo_output"
        else:
            return "dummy_output"

    def grab_energy(self, abo_file):
        pass

    def grab_piezo_tensor(self, anaddb_file=None):
        pass

    def grab_flexo_tensor(self, anaddb_file=None):
        pass

class TestPerturbations(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory.
        self.test_dir_obj = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_obj.name
        self.orig_dir = os.getcwd()
        os.chdir(self.test_dir)

        # Patch the AbinitParser so that a valid structure is returned.
        from symmstate.utils.parsers import AbinitParser
        self.parser_patch = patch(
            "symmstate.utils.parsers.AbinitParser.parse_abinit_file",
            return_value={
                "acell": [1.0, 1.0, 1.0],
                "rprim": [[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]],
                "xred": [[0.0, 0.0, 0.0]],
                "znucl": [1],
                "typat": [1]
            }
        )
        self.parser_patch.start()

        # Write a dummy Abinit file.
        dummy_content = (
            "acell 1.0 1.0 1.0\n"
            "rprim\n"
            "  1.0 0.0 0.0\n"
            "  0.0 1.0 0.0\n"
            "  0.0 0.0 1.0\n"
            "xred\n"
            "  0.0 0.0 0.0\n"
            "znucl 1\n"
            "typat 1\n"
        )
        self.dummy_abi = os.path.join(self.test_dir, "dummy.abi")
        with open(self.dummy_abi, "w") as f:
            f.write(dummy_content)

        # Create a dummy perturbation array (one atom, three coordinates).
        self.perturbation = np.ones((1, 3))
        self.batch_header = "dummy header"
        self.num_datapoints = 3

        # Instantiate the Perturbations object.
        self.pert_obj = Perturbations(
            name="test_perturb",
            num_datapoints=self.num_datapoints,
            abi_file=self.dummy_abi,
            min_amp=0.0,
            max_amp=0.5,
            perturbation=self.perturbation,
            batch_script_header_file=self.batch_header,
            host_spec="mpirun -np 1"
        )

        # Override waiting so that no actual job waiting occurs.
        self.pert_obj.abinit_file.wait_for_jobs_to_finish = lambda check_time, **kwargs: None

        # Patch the perturbations method to accept arbitrary kwargs.
        self.orig_perturbations = self.pert_obj.abinit_file.perturbations
        self.pert_obj.abinit_file.perturbations = lambda pert, **kwargs: DummyPerturbation(0)


        # Ensure the underlying structure values are set.
        self.pert_obj.abinit_file.natom = 1
        self.pert_obj.abinit_file.coordinates_xred = np.array([[0.0, 0.0, 0.0]])
        self.pert_obj.abinit_file.coordinates_xcart = np.array([[0.0, 0.0, 0.0]])

    def tearDown(self):
        os.chdir(self.orig_dir)
        self.test_dir_obj.cleanup()
        self.parser_patch.stop()
        self.pert_obj.abinit_file.perturbations = self.orig_perturbations

    def test_generate_perturbations(self):
        pert_objs = self.pert_obj.generate_perturbations()
        self.assertEqual(len(pert_objs), self.num_datapoints)
        self.assertEqual(len(self.pert_obj.list_amps), self.num_datapoints)

    def test_calculate_energy_of_perturbations(self):
        self.pert_obj.list_energies = []
        self.pert_obj.generate_perturbations()
        for p_obj in self.pert_obj.perturbed_objects:
            p_obj.run_energy_calculation("dummy")
            p_obj.grab_energy = lambda abo: None
            p_obj.running_jobs = [999]
        self.pert_obj.abinit_file.running_jobs = []
        self.pert_obj.calculate_energy_of_perturbations()
        self.assertEqual(len(self.pert_obj.list_energies), self.num_datapoints)
        for energy in self.pert_obj.list_energies:
            self.assertEqual(energy, 100)
        self.assertEqual(len(self.pert_obj.abinit_file.running_jobs), self.num_datapoints)

    def test_calculate_piezo_of_perturbations(self):
        self.pert_obj.list_energies = []
        self.pert_obj.generate_perturbations()
        for p_obj in self.pert_obj.perturbed_objects:
            p_obj.run_piezo_calculation("dummy")
            p_obj.run_mrgddb_file = lambda content: None
            p_obj.run_anaddb_file = lambda inp, piezo=False, flexo=False, peizo=False: "dummy_piezo_output"
            p_obj.running_jobs = [888]
        self.pert_obj.abinit_file.running_jobs = []
        self.pert_obj.calculate_piezo_of_perturbations()
        # Expect one energy per perturbation.
        self.assertEqual(len(self.pert_obj.list_energies), self.num_datapoints)
        for energy in self.pert_obj.list_energies:
            self.assertEqual(energy, 110)
        self.assertEqual(len(self.pert_obj.list_piezo_tensors_clamped), self.num_datapoints)
        self.assertEqual(len(self.pert_obj.list_piezo_tensors_relaxed), self.num_datapoints)

    def test_calculate_flexo_of_perturbations(self):
        self.pert_obj.list_energies = []
        self.pert_obj.generate_perturbations()
        for p_obj in self.pert_obj.perturbed_objects:
            p_obj.run_flexo_calculation("dummy")
            p_obj.run_mrgddb_file = lambda content: None
            p_obj.run_anaddb_file = lambda inp, piezo=False, flexo=False, peizo=False: "dummy_output"
            p_obj.running_jobs = [777]
        self.pert_obj.abinit_file.running_jobs = []
        self.pert_obj.calculate_flexo_of_perturbations()
        self.assertEqual(len(self.pert_obj.list_energies), self.num_datapoints)
        for energy in self.pert_obj.list_energies:
            self.assertEqual(energy, 120)
        self.assertEqual(len(self.pert_obj.list_flexo_tensors), self.num_datapoints)

    def test_clean_lists(self):
        amps = [0.1, 0.2, 0.3]
        tensors = [np.array([[1]]), None, np.array([[2]])]
        cleaned_amps, cleaned_list = self.pert_obj._clean_lists(amps, tensors)
        self.assertEqual(cleaned_amps, [0.1, 0.3])
        self.assertEqual(len(cleaned_list), 2)

    def test_record_data(self):
        self.pert_obj.abi_file = "dummy.abi"
        self.pert_obj.pert = np.array([[0.1, 0.2, 0.3]])
        self.pert_obj.list_amps = [0.1, 0.2]
        self.pert_obj.list_energies = [100, 200]
        self.pert_obj.list_piezo_tensors_clamped = [np.array([[1]]), np.array([[1.5]])]
        self.pert_obj.list_piezo_tensors_relaxed = [np.array([[2]]), np.array([[2.5]])]
        self.pert_obj.list_flexo_tensors = [np.array([[3]]), np.array([[3.5]])]
        record_file = "record.txt"
        self.pert_obj.record_data(record_file)
        with open(record_file, "r") as f:
            content = f.read()
        self.assertIn("dummy.abi", content)
        self.assertIn("100", content)
        self.assertIn("1", content)
        self.assertIn("3.5", content)

    def test_data_analysis_energy(self):
        self.pert_obj.list_amps = [0.0, 0.25, 0.5]
        self.pert_obj.list_energies = [100, 150, 200]
        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"):
            try:
                self.pert_obj.data_analysis(piezo=False, flexo=False, save_plot=False, filename="dummy_energy")
            except Exception as e:
                self.fail(f"data_analysis raised an exception in energy mode: {e}")

    def test_data_analysis_flexo(self):
        self.pert_obj.list_amps = [0.0, 0.25, 0.5]
        self.pert_obj.list_flexo_tensors = [np.array([[3]]), np.array([[3.2]]), np.array([[3.5]])]
        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"):
            try:
                self.pert_obj.data_analysis(flexo=True, save_plot=False, filename="dummy_flexo")
            except Exception as e:
                self.fail(f"data_analysis raised an exception in flexo mode: {e}")

if __name__ == "__main__":
    unittest.main()



