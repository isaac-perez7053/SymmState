import unittest
from pathlib import Path
import tempfile
from shared.boilerplate_generation import BoilerplateGenerator

class TestBoilerplateGenerator(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = Path(self.test_dir.name)

        # Mock pseudopotential directory
        self.pp_dir = self.test_dir_path / "mock_pp_path"
        self.pp_dir.mkdir()
        (self.pp_dir / "pseudo1.psp8").touch()
        (self.pp_dir / "pseudo2.psp8").touch()

        # Mock general structure file
        self.general_structure_file = self.test_dir_path / "general_structure_file.txt"
        with open(self.general_structure_file, "w") as f:
            f.write(f'pp_dirpath "{self.pp_dir}/"\n')  # Use absolute path
            f.write("ntypat 2\n")
            f.write('pseudos "pseudo1.psp8" "pseudo2.psp8"\n')

        # Mock sbatch preamble file
        self.preamble_file = self.test_dir_path / "preamble.txt"
        with open(self.preamble_file, "w") as f:
            f.write("#SBATCH --job-name=test_job\n")
            f.write("#SBATCH --output=test_output.log\n")

        # Mock input file
        self.input_file = self.test_dir_path / "example1_inputFile.in"
        with open(self.input_file, "w") as f:
            f.write(f"genstruc {self.general_structure_file}\n")
            f.write(f"sbatch_preamble {self.preamble_file}\n")


    def tearDown(self):
        # Cleanup the temporary directory
        self.test_dir.cleanup()

    def test_generate_boilerplate(self):
        # Use BoilerplateGenerator with the temporary files
        generator = BoilerplateGenerator(self.input_file)
        generator.generate_boilerplate()

        # Validate boilerplate directory is created
        boilerplate_dir = generator.target_dir
        self.assertTrue(boilerplate_dir.exists(), "Boilerplate directory was not created.")

        # Validate jobscript is created
        jobscript_path = boilerplate_dir / "jobscript.sh"
        self.assertTrue(jobscript_path.exists(), "Jobscript was not created.")

        # Validate template.abi is created
        template_path = boilerplate_dir / "template.abi"
        self.assertTrue(template_path.exists(), "Template.abi was not created.")

        # Validate pseudopotential files are copied
        for pseudo in ["pseudo1.psp8", "pseudo2.psp8"]:
            self.assertTrue((boilerplate_dir / pseudo).exists(), f"Pseudopotential {pseudo} was not copied.")


if __name__ == "__main__":
    unittest.main()

