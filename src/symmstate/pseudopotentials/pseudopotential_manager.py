import os
from typing import Dict
import argparse
import logging
from symmstate import SymmStateCore
from symmstate.utils import Logger

class PseudopotentialManager(SymmStateCore): 
    def __init__(self, folder_path: str = None, *, logger=None):
        """
        This class will initialize automatically when running SymmState
        """
        # Calculate the path to the pseudopotential folder in the SymmState package
        if folder_path is None:
            symmstate_path = self.find_package_path()
            self.folder_path = f"{symmstate_path}/pseudopotentials"
        else: 
            self.folder_path = folder_path
        
        self.logger = logger

        self.pseudo_registry: Dict[str, str] = self._load_pseudopotentials()

    def _load_pseudopotentials(self) -> Dict[str, str]:
        """ Load existing pseudopotentials from the folder into a dictionary. """
        return {name: os.path.join(self.folder_path, name) for name in os.listdir(self.folder_path)}
    
    def add_pseudopotential(self, file_path: str) -> None: 
        """Add a new pseudopotential to the folder and update the dictionary. """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        file_name = os.path.basename(file_path)
        destination = os.path.join(self.folder_path, file_name)

        # Check if file already exists in dictionary
        if file_name in self.pseudo_registry:
            Logger.log_or_print(message=f"File {file_name} already exists in the folder", logger=self.logger)

        with open(file_path, 'rb') as f_src:
            with open(destination, 'wb') as f_dest:
                f_dest.write(f_src.read())

        self.pseudo_registry[file_name] = destination
        Logger.log_or_print(message=f'Added: {file_name}', logger=self.logger)

    def get_pseudopotential(self, name: str) -> str:
        """ Retrieve the path of the pseudopotential by name. """
        return self.pseudo_registry.get(name)
    
    def delete_pseudopotential(self, name: str) -> None:
        """ Delete a pseudopotential from the folder and the dictionary. """
        if name in self.pseudo_registry:
            os.remove(self.pseudo_registry[name])
            del self.pseudo_registry[name]
            Logger.log_or_print(message=f'Deleted: {name}', logger=self.logger)
        else:
            Logger.log_or_print(message=f"Pseudopotential {name} not found when attempting to delete", logger=self.logger, level=logging.ERROR)

def main():
    manager = PseudopotentialManager()

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Manage pseudopotentials')
    # Define mutually exclusive operations: add, delete, and list.
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--add', '-a', nargs='+', help='Add pseudopotentials')
    action_group.add_argument('--delete', '-d', nargs='+', help='Delete pseudopotentials')
    action_group.add_argument('--list', '-l', action='store_true', help='List current pseudopotentials')
    
    # Parse the arguments
    args = parser.parse_args()

    if args.add:
        print("Adding pseudopotentials:")
        for pseudo_file in args.add:
            manager.add_pseudopotential(pseudo_file)
    elif args.delete:
        print("Deleting pseudopotentials:")
        for pseudo_file in args.delete:
            manager.delete_pseudopotential(pseudo_file)
    elif args.list:
        if manager.pseudo_registry:
            print("Current pseudopotentials:")
            for name, path in manager.pseudo_registry.items():
                print(f"{name} -> {path}")
        else:
            print("No pseudopotentials found.")

if __name__ == "__main__":
    main()
