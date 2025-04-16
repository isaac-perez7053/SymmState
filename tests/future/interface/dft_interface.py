# my_package/interface/base_interface.py
from abc import ABC, abstractmethod


class DFTInterface(ABC):
    """
    Base interface that must be implemented by any DFT workflow class
    (e.g., for Abinit, VASP, Quantum ESPRESSO, etc.).
    """

    @abstractmethod
    def generate_input_files(self, structure, parameters):
        """
        Create code-specific input files (POSCAR, INCAR for VASP;
        .abi for Abinit, etc.) in the correct directories,
        possibly using path utilities from utils.paths.

        :param structure: A user-defined or pymatgen structure object
                          (or whichever data structure you use to store geometry).
        :param parameters: A dictionary or object with run settings (ecut, kpoints, etc.)
        """
        pass

    @abstractmethod
    def submit_job(self, job_name: str) -> None:
        """
        Kick off a calculation for the created input files, typically by
        writing a Slurm script and calling sbatch or a direct shell command.

        :param job_name: A unique name for this job, used for filenames or job IDs.
        """
        pass

    @abstractmethod
    def parse_results(self, job_name: str) -> dict:
        """
        Parse the output files after the run completes to extract things like
        total energy, forces, band gap, or anything else relevant.

        :param job_name: The name of the completed job to parse.
        :return: Typically a dictionary with parsed results: e.g. {"energy": -123.45, "band_gap": 1.1, ...}
        """
        pass
