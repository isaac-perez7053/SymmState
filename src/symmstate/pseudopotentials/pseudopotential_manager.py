# This file is part of the SymmState package.
# This file serves as a manager for pseudopotentials used in the SymmState package.
# It allows for loading, adding, retrieving, and deleting pseudopotentials from a specified folder.
import os
from typing import Dict
import logging
from symmstate import SymmStateCore
from symmstate.utils import Logger
from symmstate.config.symm_state_settings import SymmStateSettings


class PseudopotentialManager(SymmStateCore):
    def __init__(self, *, logger=None):
        """
        This class will initialize automatically when running SymmState
        """
        # Calculate the path to the pseudopotential folder in the SymmState packag
        self.folder_path = str(SymmStateSettings().PP_DIR.resolve())

        # TODO: Possibly fix the logger to use the global logger
        self.logger = logger

        self.pseudo_registry = {}
        self.pseudo_registry: Dict[str, str] = self._load_pseudopotentials()

    def _load_pseudopotentials(self) -> Dict[str, str]:
        """Load existing pseudopotentials from the folder into a dictionary.

        Only files with allowed extensions are returned.
        """
        valid_extensions = (".psp8", ".psp", ".upf")  # adjust as needed
        pseudo_dict = {}
        for name in os.listdir(self.folder_path):
            full_path = os.path.join(self.folder_path, name)
            # Only process files (skip directories such as __pycache__)
            if os.path.isfile(full_path):
                # Skip files whose names start with __ (e.g., __init__.py)
                if name.startswith("__"):
                    continue
                # Only add files with allowed extensions.
                if name.endswith(valid_extensions):
                    pseudo_dict[name] = full_path
        return pseudo_dict

    def add_pseudopotential(self, file_path: str) -> None:
        """Add a new pseudopotential to the folder and update the dictionary."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

        file_name = os.path.basename(file_path)
        destination = os.path.join(self.folder_path, file_name)

        # Check if file already exists in dictionary
        if file_name in self.pseudo_registry:
            Logger.log_or_print(
                message=f"File {file_name} already exists in the folder",
                logger=self.logger,
            )

        with open(file_path, "rb") as f_src:
            with open(destination, "wb") as f_dest:
                f_dest.write(f_src.read())

        self.pseudo_registry[file_name] = destination
        Logger.log_or_print(message=f"Added: {file_name}", logger=self.logger)

    def get_pseudopotential(self, name: str) -> str:
        """Retrieve the path of the pseudopotential by name."""
        return self.pseudo_registry.get(name)

    def delete_pseudopotential(self, name: str) -> None:
        """Delete a pseudopotential from the folder and the dictionary."""
        if name in self.pseudo_registry:
            os.remove(self.pseudo_registry[name])
            del self.pseudo_registry[name]
            Logger.log_or_print(message=f"Deleted: {name}", logger=self.logger)
        else:
            Logger.log_or_print(
                message=f"Pseudopotential {name} not found when attempting to delete",
                logger=self.logger,
                level=logging.ERROR,
            )
