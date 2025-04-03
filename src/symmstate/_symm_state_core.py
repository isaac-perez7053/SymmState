import os
import importlib.util
import shutil
import logging
from symmstate.utils import Logger


class SymmStateCore:
    """
    Main class for symmstate package to study flexo and piezoelectricity.
    """
    # Global logger 
    _logger = Logger(name="symmstate", level=logging.INFO).logger

    def __init__(self):
        pass

    def log_or_print(
        self, 
        message: str, 
        logger: logging.Logger = None, 
        level: int = logging.INFO
    ) -> None:
        """
        Logs a message using the global logger by default.
        """
        if logger is None: 
            print(message)
        else: 
            logger = logger or self.__class__._logger  # Use class-level logger
            logger.log(level, message)
    
    @staticmethod
    def find_package_path(package_name='symmstate'):
        """Finds and returns the package path using importlib."""
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            raise ImportError(f"Cannot find package {package_name}")
        return spec.submodule_search_locations[0]
    
    @staticmethod
    def upload_files_to_package(*files, dest_folder_name):
        # Use the global logger
        logger = SymmStateCore._logger

        # Find the package path and create target directory
        package_path = SymmStateCore.find_package_path('symmstate')
        target_path = os.path.join(package_path, dest_folder_name)
        os.makedirs(target_path, exist_ok=True)

        for file in files:
            logger.info(f"Uploading file: {file}")  # Use logger.info instead of print
            
            if not os.path.isfile(file):
                logger.warning(f"File {file} does not exist.")
                continue
            
            destination_file_path = os.path.join(target_path, os.path.basename(file))
            if os.path.abspath(file) == os.path.abspath(destination_file_path):
                logger.info(f"{file} is already in {dest_folder_name}. Skipping...")
                continue

            try:
                shutil.copy(file, target_path)
                logger.info(f"Uploaded {file} to {target_path}")
            except Exception as e:
                logger.error(f"Failed to copy {file}: {e}")

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


