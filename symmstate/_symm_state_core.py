import sys
import argparse
import os
import importlib.util


class SymmStateCore:
    """
    Main class for symmstate package to study flexo and piezoelectricity.
    """

    def __init__(self):
        pass

    @staticmethod
    def find_package_path_system(package_name="symmstate"):
        """
        Finds the on-disk path of the installed package (using importlib)
        and returns both the absolute and relative paths from the current
        working directory.
        """
        # 1. Find the module spec for the package
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            raise ModuleNotFoundError(f"Cannot find module named '{package_name}'")

        # 2. The 'origin' attribute points to the "main" file for that package
        package_file = spec.origin
        if not package_file:
            raise ModuleNotFoundError(
                f"Cannot find the file for module '{package_name}'"
            )

        # 3. Get the directory containing __init__.py (or the main .py file)
        package_dir = os.path.dirname(package_file)

        # 4. Compute the relative path from the current working directory
        current_directory = os.getcwd()
        rel_path_to_package = os.path.relpath(package_dir, current_directory)

        return package_dir, rel_path_to_package

    @staticmethod
    def find_package_path(package_name="symmstate_program"):
        """
        Walks upward from the current working directory until it finds a
        directory named `package_name`. Returns the absolute path to that
        directory and the relative path from the current dir.
        """
        current_directory = os.getcwd()

        # Start from the current directory and walk upwards
        directory = current_directory
        while True:
            candidate = os.path.join(directory, package_name)
            if os.path.isdir(candidate):
                abs_path = os.path.abspath(candidate)
                rel_path = os.path.relpath(abs_path, current_directory)
                return abs_path, rel_path

            parent = os.path.dirname(directory)
            # If we have reached the top and still haven't found it:
            if parent == directory:
                raise FileNotFoundError(
                    f"Directory '{package_name}' not found above {current_directory}."
                )
            directory = parent

    @staticmethod
    def _get_unique_filename(base_name, directory="."):
        """"""
        # Get the full path for the base file
        full_path = os.path.join(directory, base_name)

        # If the file does not exist, return the base name
        if not os.path.isfile(full_path):
            return base_name

        # Otherwise, start creating new filenames with incrementing numbers
        counter = 1
        while True:
            # Format the new filename with leading zeros
            new_name = f"{os.path.splitext(base_name)[0]}_{counter:03}{os.path.splitext(base_name)[1]}"
            new_full_path = os.path.join(directory, new_name)

            # Check if the new filename is available
            if not os.path.isfile(new_full_path):
                return new_name

            # Increment the counter
            counter += 1


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
