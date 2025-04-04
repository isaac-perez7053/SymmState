from pathlib import Path
from typing import Dict
import ast

class Settings:
    SETTINGS_FILE = Path("settings.txt")

    def __init__(self):
        # If the settings file exists, load its values.
        if self.SETTINGS_FILE.exists():
            self.load_settings()
        else:
            self.set_defaults()
            self.save_settings()

    def set_defaults(self):
        # Path configurations
        self.PP_DIR: Path = Path("pseudopotentials")
        self.SMODES_PATH: Path = Path("../isobyu/smodes")
        self.WORKING_DIR: Path = Path(".")
        
        # Computational parameters
        self.DEFAULT_ECUT: int = 50  # in Hartree
        self.SYMM_PREC: float = 1e-5
        self.DEFAULT_KPT_DENSITY: float = 0.25  # k-points per Å⁻¹
        self.TEST_DIR: Path = Path("tests")
    
        # Job scheduling
        self.SLURM_HEADER: Dict = {
            "time": "24:00:00",
            "nodes": 1,
            "ntasks-per-node": 32,
            "mem": "64G"
        }
        
        # Environment-specific overrides
        self.ENVIRONMENT: str = "production"

    def load_settings(self):
        # Define the expected types for each setting.
        type_mapping = {
            "PP_DIR": Path,
            "SMODES_PATH": Path,
            "WORKING_DIR": Path,
            "DEFAULT_ECUT": int,
            "SYMM_PREC": float,
            "DEFAULT_KPT_DENSITY": float,
            "TEST_DIR": Path,
            "SLURM_HEADER": dict,
            "ENVIRONMENT": str,
        }
        with open(self.SETTINGS_FILE, "r") as f:
            for line in f:
                if not line.strip() or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key in type_mapping:
                    typ = type_mapping[key]
                    if typ == Path:
                        setattr(self, key, Path(value))
                    elif typ == dict:
                        # Use ast.literal_eval to safely evaluate the dictionary
                        setattr(self, key, ast.literal_eval(value))
                    else:
                        setattr(self, key, typ(value))
        # For any missing keys, set default values.
        for key in type_mapping:
            if not hasattr(self, key):
                self.set_defaults()
                break

    def save_settings(self):
        with open(self.SETTINGS_FILE, "w") as f:
            f.write(f"PP_DIR: {self.PP_DIR}\n")
            f.write(f"SMODES_PATH: {self.SMODES_PATH}\n")
            f.write(f"WORKING_DIR: {self.WORKING_DIR}\n")
            f.write(f"DEFAULT_ECUT: {self.DEFAULT_ECUT}\n")
            f.write(f"SYMM_PREC: {self.SYMM_PREC}\n")
            f.write(f"DEFAULT_KPT_DENSITY: {self.DEFAULT_KPT_DENSITY}\n")
            f.write(f"TEST_DIR: {self.TEST_DIR}\n")
            f.write(f"SLURM_HEADER: {self.SLURM_HEADER}\n")
            f.write(f"ENVIRONMENT: {self.ENVIRONMENT}\n")

# Create a single global instance that will be used throughout the package.
settings = Settings()

