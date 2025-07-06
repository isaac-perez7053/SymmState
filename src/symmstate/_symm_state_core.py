import os
import importlib.util
import shutil


class SymmStateCore:
    """
    Core class to SymmState
    """

    def __init__(self):
        pass

    @staticmethod
    def find_package_path(package_name="symmstate"):
        """Finds and returns the package path using importlib."""
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            raise ImportError(f"Cannot find package {package_name}")
        return spec.submodule_search_locations[0]

    @staticmethod
    def upload_files_to_package(*files, dest_folder_name):

        # Find the package path and create target directory
        package_path = SymmStateCore.find_package_path("symmstate")
        target_path = os.path.join(package_path, dest_folder_name)
        os.makedirs(target_path, exist_ok=True)

        for file in files:
            print(f"Uploading file: {file}")

            if not os.path.isfile(file):
                print(f"File {file} does not exist.")
                continue

            destination_file_path = os.path.join(target_path, os.path.basename(file))
            if os.path.abspath(file) == os.path.abspath(destination_file_path):
                print(f"{file} is already in {dest_folder_name}. Skipping...")
                continue

            try:
                shutil.copy(file, target_path)
                print(f"Uploaded {file} to {target_path}")
            except Exception as e:
                print(f"Failed to copy {file}: {e}")

        current_path = os.getcwd()
        relative_path = os.path.relpath(target_path, current_path)
        return relative_path

    @staticmethod
    def get_new_file_path(file_path, new_name):
        # Get the directory from the file path
        directory = os.path.dirname(file_path)

        # Create a new file path with the same directory and the new file name
        new_file_path = os.path.join(directory, new_name)

        return new_file_path
