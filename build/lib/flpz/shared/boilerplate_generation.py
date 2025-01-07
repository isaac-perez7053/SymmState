from pathlib import Path
import shutil

class BoilerplateGenerator:
    """
    A class to handle boilerplate generation for Abinit calculations.
    """

    def __init__(self, input_file):
        self.input_file = Path(input_file).resolve()
        # Create the boilerplate directory in the current working directory
        self.target_dir = Path.cwd() / "boilerplate"

        self.general_structure_file = None
        self.b_script_preamble = None
        self.preamble = None

    def validate_input_file(self):
        """Validate the existence of the input file."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

    def read_input_params(self):
        """Read input parameters from the input file."""
        try:
            with open(self.input_file, 'r') as f:
                lines = f.readlines()

            self.general_structure_file = next(
                line.split()[1] for line in lines if line.startswith("genstruc")
            )
            self.b_script_preamble = next(
                line.split()[1] for line in lines if line.startswith("sbatch_preamble")
            )
        except (FileNotFoundError, StopIteration):
            raise ValueError(f"Error: Missing or invalid input parameters in file: {self.input_file}")

    def setup_boilerplate(self):
        """Create boilerplate directory and copy pseudopotentials."""
        self.target_dir.mkdir(parents=True, exist_ok=True)

        general_structure_path = Path(self.general_structure_file)
        if not general_structure_path.exists():
            raise FileNotFoundError(f"General structure file not found: {self.general_structure_file}")

        with open(general_structure_path, 'r') as f:
            lines = f.readlines()

        pp_dirpath = next(
            line.split()[1].strip('"')
            for line in lines
            if line.startswith("pp_dirpath")
        )

        ntypat = int(next(line.split()[1] for line in lines if line.startswith("ntypat")))
        pseudos_line = next(line for line in lines if line.startswith("pseudos"))
        pseudos = pseudos_line.split()[1:ntypat + 1]

        for pseudo in pseudos:
            pseudo_path = Path(pp_dirpath) / pseudo.strip('"')
            if pseudo_path.exists():
                shutil.copy(pseudo_path, self.target_dir)
            else:
                print(f"Warning: Pseudopotential file not found: {pseudo_path}")

    def generate_jobscript(self, script_name="jobscript.sh", executable="abinit"):
        """Generate a jobscript in the boilerplate directory."""
        if not Path(self.b_script_preamble).exists():
            raise FileNotFoundError(f"Preamble file not found: {self.b_script_preamble}")

        with open(self.b_script_preamble, 'r') as preamble_file:
            self.preamble = preamble_file.read()

        jobscript_path = self.target_dir / script_name
        with open(jobscript_path, 'w') as script:
            script.write(f"#!/bin/bash\n{self.preamble}\n\n")
            script.write(r"mpirun -hosts=localhost -np $SLURM_NTASKS {executable} DISTNAME.abi >& log\n")

    def prepare_template(self):
        """Prepare template.abi by modifying the general structure file."""
        template_path = self.target_dir / "template.abi"
        shutil.copy(self.general_structure_file, template_path)

        with open(template_path, 'r') as f:
            lines = f.readlines()

        with open(template_path, 'w') as f:
            for line in lines:
                if line.startswith("acell"):
                    f.write("CELLDEF\n")
                elif not any(keyword in line for keyword in ["natom", "ntypat", "typat", "znucl"]):
                    f.write(line)

        # Remove block definitions like rprim, xred, xcart
        vars_to_remove = ["rprim", "xred", "xcart"]
        with open(template_path, 'r') as f:
            lines = f.readlines()

        with open(template_path, 'w') as f:
            skip = False
            for line in lines:
                if any(line.startswith(var) for var in vars_to_remove):
                    skip = True
                    continue
                if skip and line.strip() == "":
                    skip = False
                    continue
                if not skip:
                    f.write(line)

    def generate_boilerplate(self):
        """Generate boilerplate directory and files."""
        self.validate_input_file()
        self.read_input_params()
        self.setup_boilerplate()
        self.generate_jobscript()
        self.prepare_template()
        print(f"Boilerplate generation completed successfully in {self.target_dir}")


if __name__ == "__main__":
    # You can also parse an optional output directory from the command line:
    #
    #   python boilerplate_generation.py path/to/my_input.dat --out my_boilerplate_dir
    #
    import argparse

    parser = argparse.ArgumentParser(description="Generate boilerplate for Abinit calculations.")
    parser.add_argument("input_file", help="Path to the input file.")
    parser.add_argument("--out", default=None, help="Directory to generate the boilerplate in.")
    args = parser.parse_args()

    generator = BoilerplateGenerator(args.input_file, args.out)
    generator.generate_boilerplate()
