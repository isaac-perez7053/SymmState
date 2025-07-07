import json
from pathlib import Path

from symmstate.utils import find_package_path

class SymmStateSettings:
    """Handles SymmState configuration settings"""

    SETTINGS_FILE = "settings.json"
    DEFAULTS = {}

    def __init__(self):
        symmstate_path = find_package_path()

        self.DEFAULTS = {
            "PP_DIR": str(Path(symmstate_path) / "pseudopotentials"),
            "TEMPLATES_DIR": str(Path(symmstate_path) / "templates"),
            "SMODES_PATH": "../isobyu/smodes",
            "WORKING_DIR": ".",
            "PROJECT_ROOT": "/home/user/myproject",
            "DEFAULT_ECUT": 50,
            "SYMM_PREC": 1e-5,
            "TEST_DIR": "tests",
        }

        config_dir = Path(symmstate_path) / "config"
        self.SETTINGS_FILE = config_dir / "settings.json"

        self.load_settings()
        # Create the file if it doesn't exist
        if not self.SETTINGS_FILE.exists():
            self.save_settings()

    def load_settings(self):
        """Load the pre-existing settings from a json file"""
        # Start with defaults
        data = self.DEFAULTS.copy()
        # If settings file exists, update with its values
        if self.SETTINGS_FILE.exists():
            with open(self.SETTINGS_FILE, "r") as f:
                file_data = json.load(f)
                data.update(file_data)
        # Set attributes and ensure directories exist for paths
        self.PP_DIR = Path(data["PP_DIR"]).resolve()
        self.TEMPLATES_DIR = Path(data["TEMPLATES_DIR"]).resolve()
        self.SMODES_PATH = data["SMODES_PATH"]
        self.WORKING_DIR = data["WORKING_DIR"]
        self.PROJECT_ROOT = data["PROJECT_ROOT"]
        self.DEFAULT_ECUT = data["DEFAULT_ECUT"]
        self.SYMM_PREC = data["SYMM_PREC"]
        self.TEST_DIR = data["TEST_DIR"]

    def save_settings(self):
        """Saves the settings to a json file within the package"""
        data = {
            "PP_DIR": str(self.PP_DIR),
            "TEMPLATES_DIR": str(self.TEMPLATES_DIR),
            "SMODES_PATH": str(self.SMODES_PATH),
            "WORKING_DIR": str(self.WORKING_DIR),
            "DEFAULT_ECUT": self.DEFAULT_ECUT,
            "SYMM_PREC": self.SYMM_PREC,
            "TEST_DIR": str(self.TEST_DIR),
            "PROJECT_ROOT": str(self.PROJECT_ROOT),
        }
        with open(self.SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def reset_pp_dir_to_default(self):
        """Reset the pseudopotentials directory setting to its default value."""
        default_pp_dir = Path(self.DEFAULTS["PP_DIR"]).resolve()
        self.PP_DIR = default_pp_dir
        self.save_settings()


# Create a single global instance to be used throughout the package.
settings = SymmStateSettings()
