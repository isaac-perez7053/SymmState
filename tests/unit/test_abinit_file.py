import os
import shutil
import tempfile
import unittest
import numpy as np
import re
from symmstate.abinit.abinit_file import AbinitFile
from pymatgen.core import Structure, Lattice, Element

# --- Helper class to simulate a callable and subscriptable get ---
class CallableGet:
    def __init__(self, d):
        self.d = d
    def __call__(self, key, default=None):
        # Use the built-in dict.get to avoid recursion
        return dict.get(self.d, key, default)
    def __getitem__(self, key_default):
        key, default = key_default
        return dict.get(self.d, key, default)

# --- FakeVars that overrides get with a CallableGet ---
class FakeVars(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get = CallableGet(self)

# --- Dummy SlurmFile for testing ---
class DummySlurmFile:
    def __init__(self):
        self.running_jobs = []
    def write_batch_script(self, input_file, log_file, batch_name):
        # Write a dummy batch script file.
        with open(batch_name, "w") as f:
            f.write("dummy script")
        return batch_name

class TestAbinitFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files.
        self.test_dir = tempfile.mkdtemp()
        # Create a simple cubic structure with one hydrogen atom.
        lattice = Lattice.cubic(1)
        species = ["H"]
        coords = [[0, 0, 0]]
        self.dummy_structure = Structure(lattice, species, coords)
        # Create a dummy variables dictionary using FakeVars.
        data = {
            "acell": [1.0, 1.0, 1.0],
            "rprim": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
            "xred": [[0, 0, 0]],
            "xcart": [[0, 0, 0]],
            "natom": 1,
            "ntypat": 1,
            "znucl": [1],
            "typat": [1],
            "ecut": 50,
            "ecutsm": None,
            "kptrlatt": None,
            "shiftk": "0.5 0.5 0.5",
            "nband": 10,
            "nstep": 5,
            "diemac": 1000000.0,
            "ixc": 7,
            "conv_criteria": "toldfe",
            "toldfe": 0.001,
            "nshiftk": "1",
            "pseudos": ["pseudo1", "pseudo2"],
        }
        self.dummy_vars = FakeVars(data)
        # Create an AbinitFile instance using the dummy structure and a dummy SlurmFile.
        dummy_slurm = DummySlurmFile()
        self.abinit_file = AbinitFile(unit_cell=self.dummy_structure, slurm_obj=dummy_slurm)
        # Override self.vars with our fake dictionary.
        self.abinit_file.vars = self.dummy_vars
        # Set file_name to reside in the temporary directory.
        self.abinit_file.file_name = os.path.join(self.test_dir, "test_abinit")
        # Set conv_criteria on the instance so that write_custom_abifile can use it.
        self.abinit_file.conv_criteria = ("toldfe", self.dummy_vars["toldfe"])

    def tearDown(self):
        # Clean up the temporary directory after tests.
        shutil.rmtree(self.test_dir)

    def test_get_unique_filename(self):
        # Create a dummy file.
        file_path = os.path.join(self.test_dir, "testfile.txt")
        with open(file_path, "w") as f:
            f.write("content")
        unique = AbinitFile._get_unique_filename(file_path)
        # The unique filename should be different from the original file path.
        self.assertNotEqual(unique, file_path)
        # Ensure that the file with the unique filename does not exist.
        self.assertFalse(os.path.exists(unique))

    def test_write_custom_abifile(self):
        output_file = os.path.join(self.test_dir, "custom_output")
        content = "Test Header Content"
        # Ensure required keys for coordinates are present.
        self.abinit_file.vars["xred"] = [[0, 0, 0]]
        self.abinit_file.vars["xcart"] = [[0, 0, 0]]
        # Call write_custom_abifile with coords_are_cartesian set to False.
        self.abinit_file.write_custom_abifile(output_file, content, coords_are_cartesian=False)
        output_path = output_file + ".abi"
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "r") as f:
            data = f.read()
        # Verify that the header content and unit cell definitions appear.
        self.assertIn("Test Header Content", data)
        self.assertIn("acell", data)
        self.assertIn("rprim", data)
        os.remove(output_path)

    def test_grab_energy(self):
        abo_file = os.path.join(self.test_dir, "dummy.abo")
        energy_value = -123.456E+00
        with open(abo_file, "w") as f:
            f.write(f"some text\n total_energy: {energy_value}\n more text")
        self.abinit_file.grab_energy(abo_file)
        # Verify that the energy attribute is set correctly.
        self.assertTrue(hasattr(self.abinit_file, "energy"))
        np.testing.assert_almost_equal(self.abinit_file.energy, energy_value)

    def test_parse_tensor(self):
        tensor_str = """
        1.0 2.0 3.0
        4.0 5.0 6.0
        7.0 8.0 9.0
        """
        parsed = self.abinit_file.parse_tensor(tensor_str)
        expected = np.array([[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]])
        np.testing.assert_array_almost_equal(parsed, expected)

    def test_copy_abinit_file(self):
        copy_instance = self.abinit_file.copy_abinit_file()
        # Check that the copy is a different object with the same file_name and vars.
        self.assertIsNot(copy_instance, self.abinit_file)
        self.assertEqual(copy_instance.file_name, self.abinit_file.file_name)
        self.assertEqual(copy_instance.vars, self.abinit_file.vars)

    def test_grab_flexo_tensor(self):
        anaddb_file = os.path.join(self.test_dir, "dummy_flexo.abo")
        tensor_section = """TOTAL flexoelectric tensor (units= nC/m)
 xx yy zz yz xz xy
  0.1 0.2 0.3 0.4 0.5 0.6
  0.7 0.8 0.9 1.0 1.1 1.2
  1.3 1.4 1.5 1.6 1.7 1.8
  1.9 2.0 2.1 2.2 2.3 2.4
  2.5 2.6 2.7 2.8 2.9 3.0
  3.1 3.2 3.3 3.4 3.5 3.6
  3.7 3.8 3.9 4.0 4.1 4.2
  4.3 4.4 4.5 4.6 4.7 4.8
  4.9 5.0 5.1 5.2 5.3 5.4
"""
        with open(anaddb_file, "w") as f:
            f.write("Some header\n" + tensor_section + "\nSome footer")
        self.abinit_file.grab_flexo_tensor(anaddb_file)
        self.assertIsNotNone(self.abinit_file.flexo_tensor)
        self.assertEqual(self.abinit_file.flexo_tensor.shape, (9, 5))

    def test_grab_piezo_tensor(self):
        anaddb_file = os.path.join(self.test_dir, "dummy_piezo.abo")
        clamped_section = """Proper piezoelectric constants (clamped ion) (unit:c/m^2)
  0.1 0.2 0.3
  0.4 0.5 0.6
"""
        relaxed_section = """Proper piezoelectric constants (relaxed ion) (unit:c/m^2)
  0.7 0.8 0.9
  1.0 1.1 1.2
"""
        content = clamped_section + "\n" + relaxed_section
        with open(anaddb_file, "w") as f:
            f.write(content)
        self.abinit_file.grab_piezo_tensor(anaddb_file)
        expected_clamped = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        expected_relaxed = np.array([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
        np.testing.assert_array_almost_equal(self.abinit_file.piezo_tensor_clamped, expected_clamped)
        np.testing.assert_array_almost_equal(self.abinit_file.piezo_tensor_relaxed, expected_relaxed)

if __name__ == '__main__':
    unittest.main()




