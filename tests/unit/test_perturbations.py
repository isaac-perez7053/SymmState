import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Adjust these imports to match your actual paths/names
from symmstate.utils import DataParser
from symmstate.flpz.perturbations import Perturbations
from symmstate.abinit import AbinitFile
from symmstate.slurm import SlurmFile, SlurmHeader


# # DummyPerturbation to simulate a perturbed AbinitFile object (if needed)
# # But if your new code references actual classes, you can remove or modify this.
# class DummyPerturbation:
#     def __init__(self, base_value=0, slurm_obj=None):
#         self.file_name = "dummy"
#         self.slurm_obj = slurm_obj
#         self.energy = None
#         self.base_value = base_value

#     def run_energy_calculation(self, host_spec=None):
#         self.energy = 100 + self.base_value


# class TestPerturbations(unittest.TestCase):
#     def setUp(self):
#         # Create a temporary directory
#         self.test_dir_obj = tempfile.TemporaryDirectory()
#         self.test_dir = self.test_dir_obj.name
#         self.orig_dir = os.getcwd()
#         os.chdir(self.test_dir)

#         # Minimal .abi file content
#         dummy_content = """
# acell 1.0 1.0 1.0

# rprim
#   1.0 0.0 0.0
#   0.0 1.0 0.0
#   0.0 0.0 1.0

# xred
#   0.0 0.0 0.0

# xcart
#   0.0 0.0 0.0

# natom 1
# ntypat 1
# znucl 1
# typat 1

# ecut 50
# # ecutsm 0.5

# # kptrlatt
# 1 1 1
# 1 1 1
# 1 1 1

# shiftk 0.5 0.5 0.5

# nband 10
# nstep 5
# diemac 1000000.0
# ixc 7

# # conv_criteria is "toldfe", which corresponds to:
# toldfe 0.001

# nshiftk 1

# pseudos "pseudo1" "pseudo2"
# """

#         self.dummy_abi = os.path.join(self.test_dir, "dummy.abi")
#         with open(self.dummy_abi, "w") as f:
#             f.write(dummy_content)

#         self.num_datapoints = 3
#         self.min_amp = 0.0
#         self.max_amp = 0.5
#         self.perturbation = np.ones((1, 3))

#         # SlurmHeader + SlurmFile to match your HPC changes
#         self.dummy_header = SlurmHeader(
#             job_name="TestJob", partition="debug", ntasks=1, time="00:05:00"
#         )
#         self.dummy_slurm = SlurmFile(slurm_header=self.dummy_header, num_processors=1)
#         # Disable actual waiting
#         self.dummy_slurm.wait_for_jobs_to_finish = lambda *args, **kwargs: None

#         # Create the Perturbations object
#         self.pert_obj = Perturbations(
#             name="test_perturb",
#             num_datapoints=self.num_datapoints,
#             abi_file=self.dummy_abi,
#             min_amp=self.min_amp,
#             max_amp=self.max_amp,
#             perturbation=self.perturbation,
#             slurm_obj=self.dummy_slurm,
#         )

#     def tearDown(self):
#         os.chdir(self.orig_dir)
#         self.test_dir_obj.cleanup()

#     def test_generate_perturbations(self):
#         """
#         generate_perturbations() should create 'num_datapoints' AbinitFile objects
#         with distinct amplitude settings.
#         """
#         objs = self.pert_obj.generate_perturbations()
#         self.assertEqual(len(objs), self.num_datapoints)
#         self.assertEqual(len(self.pert_obj.list_amps), self.num_datapoints)

#     @patch.object(DataParser, "grab_energy")
#     def test_calculate_energy_of_perturbations(self, mock_grab_energy):
#         """
#         In the new code, calculate_energy_of_perturbations() runs energy calculations
#         for each object, but only appends ONE final energy result at the end.
#         """
#         mock_grab_energy.return_value = 123.456
#         self.pert_obj.generate_perturbations()
#         # Now loop over each generated object and set natom
#         for obj in self.pert_obj.perturbed_objects:
#             obj.vars["natom"] = 1

#         self.pert_obj.calculate_energy_of_perturbations()

#         self.assertEqual(len(self.pert_obj.results["energies"]), 1)
#         self.assertAlmostEqual(self.pert_obj.results["energies"][0], 123.456)

#     @patch.object(DataParser, "grab_energy")
#     @patch.object(DataParser, "grab_piezo_tensor")
#     def test_calculate_piezo_of_perturbations(
#         self, mock_piezo_tensor, mock_grab_energy
#     ):
#         """
#         In the new code, calculate_piezo_of_perturbations() appends one
#         energy result PER object. So we expect 'num_datapoints' energies.
#         """
#         mock_grab_energy.return_value = 210.0
#         mock_piezo_tensor.return_value = ("dummy_clamped", "dummy_relaxed")

#         self.pert_obj.generate_perturbations()
#         self.pert_obj.calculate_piezo_of_perturbations()

#         # energies and piezo data should match num_datapoints
#         self.assertEqual(len(self.pert_obj.results["energies"]), self.num_datapoints)
#         self.assertEqual(
#             len(self.pert_obj.results["piezo"]["clamped"]), self.num_datapoints
#         )
#         self.assertEqual(
#             len(self.pert_obj.results["piezo"]["relaxed"]), self.num_datapoints
#         )

#         # All energies should be 210.0
#         for energy in self.pert_obj.results["energies"]:
#             self.assertEqual(energy, 210.0)

#     @patch.object(DataParser, "grab_energy")
#     @patch.object(DataParser, "grab_flexo_tensor")
#     @patch.object(DataParser, "grab_piezo_tensor")
#     def test_calculate_flexo_of_perturbations(
#         self, mock_piezo_tensor, mock_flexo_tensor, mock_grab_energy
#     ):
#         """
#         In the new code, calculate_flexo_of_perturbations() also appends
#         one result PER object: energies, flexo, plus piezo clamped/relaxed.
#         """
#         mock_grab_energy.return_value = 300.0
#         mock_flexo_tensor.return_value = np.array([[3.0]])
#         mock_piezo_tensor.return_value = ("clamped_dummy", "relaxed_dummy")

#         self.pert_obj.generate_perturbations()
#         self.pert_obj.calculate_flexo_of_perturbations()

#         # Should have 3 energies, 3 flexo, 3 clamped, 3 relaxed
#         self.assertEqual(len(self.pert_obj.results["energies"]), self.num_datapoints)
#         self.assertEqual(len(self.pert_obj.results["flexo"]), self.num_datapoints)
#         self.assertEqual(
#             len(self.pert_obj.results["piezo"]["clamped"]), self.num_datapoints
#         )
#         self.assertEqual(
#             len(self.pert_obj.results["piezo"]["relaxed"]), self.num_datapoints
#         )

#         # All energies should be 300.0
#         for energy in self.pert_obj.results["energies"]:
#             self.assertEqual(energy, 300.0)

#     def test_record_data(self):
#         """
#         record_data() should write the key fields and results to a file.
#         """
#         record_file = "record.txt"
#         self.pert_obj.abi_file = "dummy.abi"
#         self.pert_obj.pert = np.array([[0.1, 0.2, 0.3]])
#         self.pert_obj.list_amps = [0.1, 0.2]
#         self.pert_obj.results["energies"] = [100, 200]
#         self.pert_obj.results["piezo"]["clamped"] = [np.array([[1]]), np.array([[1.5]])]
#         self.pert_obj.results["piezo"]["relaxed"] = [np.array([[2]]), np.array([[2.5]])]
#         self.pert_obj.results["flexo"] = [np.array([[3]]), np.array([[3.5]])]

#         self.pert_obj.record_data(record_file)
#         with open(record_file, "r") as f:
#             content = f.read()

#         self.assertIn("dummy.abi", content)
#         self.assertIn("0.1 0.2 0.3", content)  # from self.pert_obj.pert
#         self.assertIn("List of Amplitudes: [0.1, 0.2]", content)
#         self.assertIn("energies: [100, 200]", content)
#         self.assertIn(
#             "piezo: {'clamped': [array([[1]]), array([[1.5]])], 'relaxed': [array([[2]]), array([[2.5]])]}",
#             content,
#         )
#         self.assertIn("flexo: [array([[3]]), array([[3.5]])]", content)

#     @patch("matplotlib.pyplot.show")
#     @patch("matplotlib.pyplot.savefig")
#     def test_data_analysis_energy(self, mock_savefig, mock_show):
#         """
#         data_analysis() in default mode plots energies vs. amplitude.
#         """
#         self.pert_obj.list_amps = [0.0, 0.25, 0.5]
#         self.pert_obj.results["energies"] = [100, 150, 200]

#         try:
#             self.pert_obj.data_analysis(
#                 piezo=False, flexo=False, save_plot=False, filename="dummy_energy"
#             )
#         except Exception as e:
#             self.fail(f"data_analysis raised an exception in energy mode: {e}")

#     @patch("matplotlib.pyplot.show")
#     @patch("matplotlib.pyplot.savefig")
#     def test_data_analysis_flexo(self, mock_savefig, mock_show):
#         """
#         data_analysis() in flexo mode plots flexoelectric tensors vs amplitude.
#         """
#         self.pert_obj.list_amps = [0.0, 0.25, 0.5]
#         self.pert_obj.results["flexo"] = [
#             np.array([[3.0]]),
#             np.array([[3.2]]),
#             np.array([[3.5]]),
#         ]
#         try:
#             self.pert_obj.data_analysis(
#                 flexo=True, save_plot=False, filename="dummy_flexo"
#             )
#         except Exception as e:
#             self.fail(f"data_analysis raised an exception in flexo mode: {e}")


# if __name__ == "__main__":
#     unittest.main()
