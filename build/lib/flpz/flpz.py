#!/usr/bin/env python3
import re
import sys
import argparse
import os
from pathlib import Path


# Add subdirectories to the Python path for importing submodules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'energy'))
sys.path.append(str(current_dir / 'perturbations'))
sys.path.append(str(current_dir / 'coupling'))


class flpz:
    """
    Main class for FLPZ package to study flexo and piezoelectricity.
    """

    def __init__(self, input_file, smodes_input, target_irrep):
        self.input_file = input_file
        self.smodes_input = smodes_input
        self.target_irrep = target_irrep

        self.name = None
        self.genstruc = None
        self.num_datapoints = None
        self.min_amp = None
        self.max_amp = None
        self.batch_header_file = None

        self._parse_inputfile()

    def _parse_inputfile(self):
        """
        Returns the variables contained in the flpz input file
        """


        # Check if the input file exists
        if not os.path.isfile(self.input_file):
            raise FileNotFoundError(f"The input file '{self.input_file}' does not exist.")

        with open(self.input_file, 'r') as f:
            lines = f.readlines()


        # Extract num_datapoints
        num_datapoints = None
        for i, line in enumerate(lines):
            if line.strip().startswith('num_datapoints'):
                match = re.search(r"\d+", line)
                if match:
                    num_datapoints = int(match.group())
                    break  # Exit the loop once 'num_datapoints' is found

        self.num_datapoints = num_datapoints
        # If num_datapoints is not set, raise an exception
        if self.num_datapoints is None:
            raise Exception("Number of datapoints is missing in the input file!")

        # Extract name
        name = None
        for line in lines:
            if line.strip().startswith('name'):
                match = re.search(r"name\s+([a-zA-Z0-9_.-]+)", line)
                if match:
                    name = str(match.group(1))  # Get the alphanumeric value after 'name'
                    break
        
        self.name = name
        if self.name is None:
            raise Exception("Name is missing in the input file!")
            
        # Extract genstruc abinit file name
        genstruc = None
        for line in lines:
            if line.strip().startswith('genstruc'):
                match = re.search(r"genstruc\s+([a-zA-Z0-9_.\-/\\]+)", line)
                if match:
                    genstruc = str(match.group(1)) # Capture the filename after 'genstruc'
                    break

        self.genstruc = genstruc
        if self.genstruc is None:
            raise Exception("The Abinit file (genstruc) is missing in the input file!")

            
        # Extract the minimum amplitude
        min_amp = None
        for line in lines:
            if line.strip().startswith('min'):
                match = re.search(r"min\s+([-+]?\d*\.?\d+)", line)
                if match:
                    min_amp = float(match.group(1))  # Get the numeric value after 'min'
                    break

        self.min_amp = min_amp
        if self.min_amp is None:
            raise Exception("The min_amp (min) is missing in the input file!")

        # Extract the maximum amplitude
        max_amp = None
        for line in lines:
            if line.strip().startswith('max'):
                match = re.search(r"max\s+([-+]?\d*\.?\d+)", line)
                if match:
                    max_amp = float(match.group(1))  # Convert the numeric value to float
                    break

        self.max_amp = max_amp
        if self.max_amp is None:
            raise Exception("The maximum amplitude ('max') is missing or not found in the input file!")
            
        # Extract the sbatch preamble
        sbatch_preamble = None
        for line in lines:
            if line.strip().startswith('sbatch_preamble'):
                match = re.search(r"sbatch_preamble\s+([a-zA-Z0-9_.\-/\\]+)", line)
                if match:
                    sbatch_preamble = str(match.group(1))  # Capture the filename after 'sbatch_preamble'
                    break

        self.batch_header_file = sbatch_preamble
        if sbatch_preamble is None:
            raise Exception("sbatch preamble is missing in the input file!")


def main():
    parser = argparse.ArgumentParser(description='FLPZ: A tool to study flexo and piezoelectricity')
    parser.add_argument('inputs', nargs='*', help='Program type (energy, pert, couple) followed by their necessary arguments')

    args = parser.parse_args()
    inputs = args.inputs

    # Validate inputs
    if len(inputs) == 0:
        print("Error: No program type was specified.")
        sys.exit(1)

    program_type = inputs[0]
    additional_args = inputs[1:]  # The remaining arguments

    # Instantiate FLPZ class
    flpz_instance = flpz()

    # Dispatch the appropriate method based on program type
    if program_type == 'energy':
        flpz_instance.energy(*additional_args)
    elif program_type == 'pert':
        flpz_instance.perturbations(*additional_args)
    elif program_type == 'couple':
        flpz_instance.coupling(*additional_args)
    else:
        print(f"Error: Invalid program type '{program_type}'. Available options are: energy, pert, couple.")


if __name__ == "__main__":
    main()