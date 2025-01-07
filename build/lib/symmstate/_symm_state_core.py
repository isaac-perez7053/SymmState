import sys
import argparse
import os
from pathlib import Path
import importlib.util


class SymmStateCore:
    """
    Main class for FLPZ package to study flexo and piezoelectricity.
    """

    def __init__(self):
        pass

    def find_package_path(package_name="flpz"):
        """Calculates the path to the specified package."""

        current_directory = os.getcwd()
        spec = importlib.util.find_spec(package_name)

        if spec and spec.submodule_search_locations:
            # The first entry in submodule_search_locations is typically the package path
            rel_path_to_package = os.path.relpath(spec, current_directory)
            return spec.submodule_search_locations[0], rel_path_to_package
        else:
            print("Package not found")


def main():
    parser = argparse.ArgumentParser(
        description="FLPZ: A tool to study flexo and piezoelectricity"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Program type (energy, pert, couple) followed by their necessary arguments",
    )

    args = parser.parse_args()
    inputs = args.inputs

    # Validate inputs
    if len(inputs) == 0:
        print("Error: No program type was specified.")
        sys.exit(1)

    program_type = inputs[0]
    additional_args = inputs[1:]  # The remaining arguments

    # Instantiate FLPZ class
    flpz_instance = SymmStateCore()

    # Dispatch the appropriate method based on program type
    if program_type == "energy":
        flpz_instance.energy(*additional_args)
    elif program_type == "pert":
        flpz_instance.perturbations(*additional_args)
    elif program_type == "couple":
        flpz_instance.coupling(*additional_args)
    else:
        print(
            f"Error: Invalid program type '{program_type}'. Available options are: energy, pert, couple."
        )


if __name__ == "__main__":
    main()
