import os
from symmstate.config import symm_state_settings


settings = symm_state_settings.settings


def get_project_root():
    return settings.PROJECT_ROOT


def get_data_path(*segments):
    """
    Joins PROJECT_ROOT/data with the given subpaths (segments).
    E.g. get_data_path("mysim", "input.in")
         => "/home/username/my_hpc_project/data/mysim/input.in"
    """
    return os.path.join(settings.PROJECT_ROOT, "data", *segments)


def get_scripts_path(*segments):
    """Joins PROJECT_ROOT/scripts with the given subpaths."""
    return os.path.join(settings.PROJECT_ROOT, "scripts", *segments)


def get_logs_path(*segments):
    """Joins PROJECT_ROOT/logs with the given subpaths."""
    return os.path.join(settings.PROJECT_ROOT, "logs", *segments)


def get_temp_path(*segments):
    """Joins PROJECT_ROOT/temp with the given subpaths."""
    return os.path.join(settings.PROJECT_ROOT, "temp", *segments)


def get_results_path(*segments):
    return os.path.join(settings.PROJECT_ROOT, "results", *segments)
