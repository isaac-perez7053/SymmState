from pathlib import Path
from typing import Dict

class Settings():
    # Path configurations
    PP_DIR: Path = Path("pseudopotentials")
    SMODES_PATH: Path = Path("../isobyu/smodes")
    WORKING_DIR: Path = Path(".")
    
    # Computational parameters
    DEFAULT_ECUT: int = 50  # in Hartree
    SYMM_PREC: float = 1e-5
    DEFAULT_KPT_DENSITY: float = 0.25  # k-points per Å⁻¹
    
    # Job scheduling
    SLURM_HEADER: Dict = {
        "time": "24:00:00",
        "nodes": 1,
        "ntasks-per-node": 32,
        "mem": "64G"
    }
    
    # Environment-specific overrides
    ENVIRONMENT: str = "production"
    
    class Config:
        env_prefix = "SYMMSTATE_"
        env_file = ".env"
        env_file_encoding = "utf-8"
    

# Singleton configuration instance
settings = Settings()