import os
import importlib.util
import shutil


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
    def find_package_path(package_name='symmstate'):
        """Finds and returns the package path using importlib."""
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            raise ImportError(f"Cannot find package {package_name}")
        return spec.submodule_search_locations[0]
    
    @staticmethod
    def upload_files_to_package(*files, dest_folder_name):
        """Uploads files to a specified directory within a package from a single input string."""

        # Find the package path and create target directory path
        package_path = SymmStateCore.find_package_path('symmstate')
        target_path = os.path.join(package_path, dest_folder_name)

        # Ensure target directory exists
        os.makedirs(target_path, exist_ok=True)

        # Copy files to target directory
        for file in files:
            print(f"Uploading file: {file}")
            
            if not os.path.isfile(file):
                print(f"File {file} does not exist.")
                continue
            
            # Define the full path of the target file
            destination_file_path = os.path.join(target_path, os.path.basename(file))
            
            # Check if source file is the same as destination file
            if os.path.abspath(file) == os.path.abspath(destination_file_path):
                print(f"{file} is already in {dest_folder_name}. Skipping...")
                continue

            # Attempt to copy the file
            try:
                shutil.copy(file, target_path)
                print(f"Uploaded {file} to {target_path}")
            except Exception as e:
                print(f"Failed to copy {file}: {e}")

        # Calculate and return relative path
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


