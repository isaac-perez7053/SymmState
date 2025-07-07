import os
from typing import Dict, Optional
import json
import shutil
from pathlib import Path

from symmstate.config import settings
from symmstate import SymmStateCore


class TemplateManager(SymmStateCore):
    """
    Manages creation, lookup, and maintenance of Abinit template files.

    Attributes:
        SPECIAL_FILE (str):
            Name of the JSON file used to record “special” templates.
        folder_path (str):
            Filesystem directory under which all templates live.
        logger (Optional[logging.Logger]):
            Logger for diagnostic or debug messages.
        template_registry (Dict[str, str]):
            Maps template filenames (e.g. "run.abi") to their full paths.
        special_templates (Dict[str, str]):
            Maps user‐defined roles (e.g. "energy") to a template filename.
    """

    SPECIAL_FILE = "special_templates.json"

    def __init__(self):
        """
        Initialize the TemplateManager by loading existing and special templates.

        Parameters:
            None:
        """
        self.folder_path = str(settings.TEMPLATES_DIR)
        self.template_registry = {}
        self.template_registry: Dict[str, str] = self._load_existing_templates()
        self.special_templates = self._load_special_templates()

    def _load_existing_templates(self):
        """
        Scan the template directory and build the registry for all “.abi” files.

        Returns:
            Dict[str, str]: Mapping from each ".abi" filename to its absolute path.
        """
        template_registry = {}
        if not os.path.exists(self.folder_path):
            return

        for file_name in os.listdir(self.folder_path):
            if str(Path(file_name).suffix) == ".abi":
                template_registry[file_name] = os.path.join(self.folder_path, file_name)
        return template_registry

    def add_template(self, template_file: str) -> str:
        """
        Copy a new .abi template into the registry, validating its name.

        Parameters:
            template_file (str):
                Path to the source .abi file to add.

        Returns:
            str: Full path to the newly installed template in folder_path.

        Raises:
            FileExistsError: If a template with the same name already exists.
        """
        # Validate template name
        template_name = os.path.basename(template_file)
        if not template_name.endswith(".abi"):
            template_name += ".abi"

        if self.template_exists(template_name):
            print(f"Template '{template_name}' already exists silly")

        # Create full output path
        output_path = os.path.join(self.folder_path, template_name)

        # Copy the file to the template directory
        shutil.copyfile(template_file, output_path)

        # Add to registry
        self.template_registry[template_name] = output_path

        return output_path

    def template_exists(self, template_file: str) -> bool:
        """
        Check whether a template is already registered or on disk.

        Parameters:
            template_file (str):
                Filename of the template to check (e.g. "run.abi").

        Returns:
            bool: True if the template exists in registry or filesystem.
        """
        exists_in_registry = template_file in self.template_registry
        exists_in_fs = os.path.exists(os.path.join(self.folder_path, template_file))
        return exists_in_registry or exists_in_fs

    def get_template_path(self, template_file: str) -> Optional[str]:
        """
        Retrieve the full path of a registered template.

        Parameters:
            template_file (str):
                Filename of the template.

        Returns:
            Optional[str]: Full filesystem path or None if not found.
        """
        return self.template_registry.get(template_file)

    def remove_template(self, template_file: str):
        """
        Delete a template from both registry and disk.

        Parameters:
            template_file (str):
                Filename of the template to remove.

        Raises:
            KeyError: If the template is not registered.
        """
        if template_file in self.template_registry:
            os.remove(self.template_registry[template_file])
            del self.template_registry[template_file]

    def _load_special_templates(self):
        """
        Load the JSON file of special templates into memory.

        Returns:
            Dict[str, str]: Mapping from role names to template filenames.
        """
        path = os.path.join(self.folder_path, self.SPECIAL_FILE)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def set_special_template(self, role: str, template_file: str):
        """
        Assign a template to a special role and persist the mapping.

        Parameters:
            role (str): Identifier for the special template (e.g. "dielectric").
            template_file (str): Filename of an existing template.

        Raises:
            FileNotFoundError: If the named template_file does not exist.
        """
        self.special_templates[role] = template_file
        with open(os.path.join(self.folder_path, self.SPECIAL_FILE), "w") as f:
            json.dump(self.special_templates, f, indent=2)

    def get_special_template_name(self, role: str) -> Optional[str]:
        return self.special_templates.get(role)

    def unload_special_template(self, role: str) -> str:
        """
        Look up which template filename is assigned to a given role.

        Parameters:
            role (str): The special template role.

        Returns:
            Optional[str]: Filename of the special template or None if unset.
        """
        template_path = self.get_special_template_name(role)
        if template_path and os.path.exists(template_path):
            with open(template_path, "r") as f:
                return f.read()
        raise FileNotFoundError(f"Special template for '{role}' not found.")

    def unload_template(self, template_file: str) -> str:
        """
        Load and return the contents of a role‐assigned special template.

        Parameters:
            role (str): The special template role to unload.

        Returns:
            str: The full file content as a string.

        Raises:
            FileNotFoundError: If no template is registered for that role or file is missing.
        """
        template_path = self.get_template_path(template_file)
        if template_path and os.path.exists(template_path):
            with open(template_path, "r") as f:
                return f.read()
        raise FileNotFoundError(f"Template '{template_file}' not found.")

    def unload_special_template(self, role: str) -> str:
        """
        Load and return the contents of a standard template file.

        Parameters:
            template_file (str): Filename of the template to unload.

        Returns:
            str: The full file content as a string.

        Raises:
            FileNotFoundError: If the template is not registered or missing on disk.
        """
        template_name = str(self.get_special_template_name(role))
        template_path = self.get_template_path(template_name)
        if template_path and os.path.exists(template_path):
            with open(template_path, "r") as f:
                return f.read()
        raise FileNotFoundError(f"Special template for '{role}' not found.")

    def is_special_template(self, template_file: str) -> bool:
        """
        Determine if a given template is marked as “special.”

        Parameters:
            template_file (str): Filename of the template.

        Returns:
            bool: True if the file appears in special_templates values.
        """
        return template_file in self.special_templates.values()

    def delete_special_template(self, role: str):
        """
        Remove a special‐role assignment and update the JSON mapping.

        Parameters:
            role (str): The special template role to delete.

        Raises:
            KeyError: If the role is not present in special_templates.
        """
        if role in self.special_templates:
            del self.special_templates[role]
            with open(os.path.join(self.folder_path, self.SPECIAL_FILE), "w") as f:
                json.dump(self.special_templates, f, indent=2)
        else:
            raise KeyError(f"No special template found for role '{role}'")
