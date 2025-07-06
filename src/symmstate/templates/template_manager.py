import os
from typing import Dict, Optional
from symmstate import SymmStateCore
import json
from symmstate.config.symm_state_settings import SymmStateSettings
import shutil
from pathlib import Path


class TemplateManager(SymmStateCore):
    """Manages creation and tracking of Abinit template files"""

    SPECIAL_FILE = "special_templates.json"

    def __init__(self, *, logger=None):
        """
        This class will initialize automatically when running SymmState
        """
        self.folder_path = str(SymmStateSettings().TEMPLATES_DIR)

        # TODO: Possibly fix the logger to use the global logger
        self.logger = logger

        self.template_registry = {}
        self.template_registry: Dict[str, str] = self._load_existing_templates()
        self.special_templates = self._load_special_templates()

    def _load_existing_templates(self):
        """Load existing templates into registry on initialization"""
        template_registry = {}
        if not os.path.exists(self.folder_path):
            return

        for file_name in os.listdir(self.folder_path):
            if str(Path(file_name).suffix) == ".abi":
                template_registry[file_name] = os.path.join(self.folder_path, file_name)
        return template_registry

    def add_template(self, template_file: str) -> str:
        """
        Add a template file to the template directory with validation.
        Returns path to created template.
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
        """Check if template exists in registry or filesystem"""
        exists_in_registry = template_file in self.template_registry
        exists_in_fs = os.path.exists(os.path.join(self.folder_path, template_file))
        return exists_in_registry or exists_in_fs

    def get_template_path(self, template_file: str) -> Optional[str]:
        """Get full path for a template by name"""
        return self.template_registry.get(template_file)

    def remove_template(self, template_file: str):
        """Remove template from registry and filesystem"""
        if template_file in self.template_registry:
            os.remove(self.template_registry[template_file])
            del self.template_registry[template_file]

    def _load_special_templates(self):
        path = os.path.join(self.folder_path, self.SPECIAL_FILE)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def set_special_template(self, role: str, template_file: str):
        self.special_templates[role] = template_file
        with open(os.path.join(self.folder_path, self.SPECIAL_FILE), "w") as f:
            json.dump(self.special_templates, f, indent=2)

    def get_special_template_name(self, role: str) -> Optional[str]:
        return self.special_templates.get(role)

    def unload_special_template(self, role: str) -> str:
        """
        Load the contents of a special template (e.g., 'energy') and return as a multiline string.
        """
        template_path = self.get_special_template_name(role)
        if template_path and os.path.exists(template_path):
            with open(template_path, "r") as f:
                return f.read()
        raise FileNotFoundError(f"Special template for '{role}' not found.")

    def unload_template(self, template_file: str) -> str:
        """
        Load the contents of a template file and return as a multiline string.
        """
        template_path = self.get_template_path(template_file)
        if template_path and os.path.exists(template_path):
            with open(template_path, "r") as f:
                return f.read()
        raise FileNotFoundError(f"Template '{template_file}' not found.")

    def unload_special_template(self, role: str) -> str:
        """
        Load the contents of a special template and return as a multiline string.
        """
        template_name = str(self.get_special_template_name(role))
        template_path = self.get_template_path(template_name)
        if template_path and os.path.exists(template_path):
            with open(template_path, "r") as f:
                return f.read()
        raise FileNotFoundError(f"Special template for '{role}' not found.")

    def is_special_template(self, template_file: str) -> bool:
        """
        Check if a template file is a special template.
        """
        return template_file in self.special_templates.values()

    def delete_special_template(self, role: str):
        """
        Delete a special template by its role.
        """
        if role in self.special_templates:
            del self.special_templates[role]
            with open(os.path.join(self.folder_path, self.SPECIAL_FILE), "w") as f:
                json.dump(self.special_templates, f, indent=2)
        else:
            raise KeyError(f"No special template found for role '{role}'")
