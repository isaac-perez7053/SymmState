import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from symmstate.flpz.perturbations import Perturbations
from symmstate.slurm_file import SlurmFile

# DummyPerturbation simulates a perturbed AbinitFile object.
class DummyPerturbation:
    def __init__(self, base_value=0, slurm_obj=None):
        self.file_name = "dummy"
        # Set the slurm_obj (provided by the test)
        self.slurm_obj = slurm_obj
        # Initialize running_jobs in the slurm_obj (this list will be overwritten in tests)
        self.slurm_obj.running_jobs = [123]
        self.energy = None
        self.piezo_tensor_clamped = None
        self.piezo_tensor_relaxed = None
        self.flexo_tensor = None
        self.base_value = base_value

    def run_energy_calculation(self, host_spec=None):
        self.energy = 100 + self.base_value

    def run_piezo_calculation(self, host_spec=None):
        self.energy = 110 + self.base_value
        self.piezo_tensor_clamped = np.array([[1 + self.base_value]])
        self.piezo_tensor_relaxed = np.array([[2 + self.base_value]])

    def run_flexo_calculation(self, host_spec=None):
        self.energy = 120 + self.base_value
        self.flexo_tensor = np.array([[3 + self.base_value]])

    def run_mrgddb_file(self, content):
        pass

    def run_anaddb_file(self, inp, piezo=False, flexo=False, peizo=False):
        if flexo:
            return "dummy_flexo_output"
        elif peizo:
            return "dummy_piezo_output"
        else:
            return "dummy_output"

    def grab_energy(self, abo):
        pass

    def grab_piezo_tensor(self, anaddb_file=None):
        pass

    def grab_flexo_tensor(self, anaddb_file=None):
        pass

class TestNewPerturbations(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for tests.
        self.test_dir_obj = tempfile.TemporaryDirectory()
        self.test_dir = self.test_dir_obj.name
        self.orig_dir = os.getcwd()
        os.chdir(self.test_dir)

        # Create a dummy Abinit file with convergence criteria.
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
            "toldfe 1.0e-6\n"
        )
        self.dummy_abi = os.path.join(self.test_dir, "dummy.abi")
        with open(self.dummy_abi, "w") as f:
            f.write(dummy_content)

        self.perturbation = np.ones((1, 3))
        self.num_datapoints = 3

        # Create a dummy SlurmFile instance.
        self.dummy_slurm = SlurmFile("dummy header\n#SBATCH --time=00:05:00", num_processors=1)
        self.dummy_slurm.wait_for_jobs_to_finish = lambda check_time, **kwargs: None

        # Patch AbinitFile.perturbations to always return a DummyPerturbation with our dummy_slurm.
        from symmstate.abinit import AbinitFile
        dummy_slurm = self.dummy_slurm  # capture in local variable
        self.orig_perturbations = AbinitFile.perturbations
        AbinitFile.perturbations = lambda abinit_self, pert, **kwargs: DummyPerturbation(0, slurm_obj=dummy_slurm)

        # Initialize Perturbations with the dummy SlurmFile.
        self.pert_obj = Perturbations(
            name="test_perturb",
            num_datapoints=self.num_datapoints,
            abi_file=self.dummy_abi,
            min_amp=0.0,
            max_amp=0.5,
            perturbation=self.perturbation,
            slurm_obj=self.dummy_slurm
        )

        # Set dummy structure values.
        self.pert_obj.abinit_file.natom = 1
        self.pert_obj.abinit_file.coordinates_xred = np.array([[0.0, 0.0, 0.0]])
        self.pert_obj.abinit_file.coordinates_xcart = np.array([[0.0, 0.0, 0.0]])
        # Patch the wait method on the AbinitFile to use the dummy slurm's wait.
        self.pert_obj.abinit_file.wait_for_jobs_to_finish = self.dummy_slurm.wait_for_jobs_to_finish

    def tearDown(self):
        os.chdir(self.orig_dir)
        self.test_dir_obj.cleanup()
        from symmstate.abinit import AbinitFile
        AbinitFile.perturbations = self.orig_perturbations

    def test_generate_perturbations(self):
        objs = self.pert_obj.generate_perturbations()
        self.assertEqual(len(objs), self.num_datapoints)
        self.assertEqual(len(self.pert_obj.list_amps), self.num_datapoints)

    def test_calculate_energy_of_perturbations(self):
        self.pert_obj.generate_perturbations()
        for obj in self.pert_obj.perturbed_objects:
            obj.run_energy_calculation(host_spec=self.dummy_slurm)
            obj.grab_energy = lambda abo: None
        # Ensure dummy_slurm.running_jobs is non-empty.
        self.dummy_slurm.running_jobs = [888]
        self.pert_obj.calculate_energy_of_perturbations()
        self.assertEqual(len(self.pert_obj.results["energies"]), self.num_datapoints)
        for energy in self.pert_obj.results["energies"]:
            self.assertEqual(energy, 100)

    def test_calculate_piezo_of_perturbations(self):
        self.pert_obj.generate_perturbations()
        for obj in self.pert_obj.perturbed_objects:
            obj.run_piezo_calculation(host_spec=self.dummy_slurm)
            obj.run_mrgddb_file = lambda content: None
            obj.run_anaddb_file = lambda inp, piezo=False, flexo=False, peizo=False: "dummy_piezo_output"
            obj.grab_energy = lambda abo: None
            obj.grab_piezo_tensor = lambda anaddb_file=None: None
            # Set each DummyPerturbation's slurm_obj.running_jobs to a dummy value.
            obj.slurm_obj.running_jobs = [888]
        self.dummy_slurm.running_jobs = [888]
        self.pert_obj.calculate_piezo_of_perturbations()
        self.assertEqual(len(self.pert_obj.results["energies"]), self.num_datapoints)
        for energy in self.pert_obj.results["energies"]:
            self.assertEqual(energy, 110)
        self.assertEqual(len(self.pert_obj.results["piezo"]["clamped"]), self.num_datapoints)
        self.assertEqual(len(self.pert_obj.results["piezo"]["relaxed"]), self.num_datapoints)

    def test_calculate_flexo_of_perturbations(self):
        self.pert_obj.generate_perturbations()
        for obj in self.pert_obj.perturbed_objects:
            obj.run_flexo_calculation(host_spec=self.dummy_slurm)
            obj.run_mrgddb_file = lambda content: None
            obj.run_anaddb_file = lambda inp, piezo=False, flexo=False, peizo=False: "dummy_output"
            obj.grab_energy = lambda abo: None
            obj.grab_flexo_tensor = lambda anaddb_file=None: None
            obj.grab_piezo_tensor = lambda anaddb_file=None: None
            obj.slurm_obj.running_jobs = [777]
        self.dummy_slurm.running_jobs = [777]
        self.pert_obj.calculate_flexo_of_perturbations()
        self.assertEqual(len(self.pert_obj.results["energies"]), self.num_datapoints)
        for energy in self.pert_obj.results["energies"]:
            self.assertEqual(energy, 120)
        self.assertEqual(len(self.pert_obj.results["flexo"]), self.num_datapoints)

    def test_record_data(self):
        self.pert_obj.abi_file = "dummy.abi"
        self.pert_obj.pert = np.array([[0.1, 0.2, 0.3]])
        self.pert_obj.list_amps = [0.1, 0.2]
        self.pert_obj.results["energies"] = [100, 200]
        self.pert_obj.results["piezo"]["clamped"] = [np.array([[1]]), np.array([[1.5]])]
        self.pert_obj.results["piezo"]["relaxed"] = [np.array([[2]]), np.array([[2.5]])]
        self.pert_obj.results["flexo"] = [np.array([[3]]), np.array([[3.5]])]
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
        self.pert_obj.results["energies"] = [100, 150, 200]
        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"):
            try:
                self.pert_obj.data_analysis(piezo=False, flexo=False, save_plot=False, filename="dummy_energy")
            except Exception as e:
                self.fail(f"data_analysis raised an exception in energy mode: {e}")

    def test_data_analysis_flexo(self):
        self.pert_obj.list_amps = [0.0, 0.25, 0.5]
        self.pert_obj.results["flexo"] = [np.array([[3]]), np.array([[3.2]]), np.array([[3.5]])]
        with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"):
            try:
                self.pert_obj.data_analysis(flexo=True, save_plot=False, filename="dummy_flexo")
            except Exception as e:
                self.fail(f"data_analysis raised an exception in flexo mode: {e}")

if __name__ == "__main__":
    unittest.main()





