from symmstate.config.symm_state_settings import settings
from symmstate.flpz.smodes_processor import SmodesProcessor
from symmstate.flpz.perturbations import Perturbations
from symmstate.flpz import FlpzCore
from symmstate.slurm import SlurmFile


class EnergyProgram(FlpzCore):
    """
    EnergyProgram runs a series of SMODES and perturbation calculations to analyze
    energy, piezoelectric, and flexoelectric properties.
    """

    def __init__(
        self,
        name: str = None,
        num_datapoints: int = None,
        abi_file: str = None,
        min_amp: int = None,
        max_amp: int = None,
        smodes_input: str = None,
        target_irrep: str = None,
        slurm_obj: SlurmFile = None,
        disp_mag: float = 0.001,
        unstable_threshold: float = -20,
    ):
        # Initialize the FlpzCore superclass.
        super().__init__(
            name=name,
            num_datapoints=num_datapoints,
            abi_file=abi_file,
            min_amp=min_amp,
            max_amp=max_amp,
        )

        self.smodes_input = smodes_input
        self.target_irrep = target_irrep
        self.slurm_obj = slurm_obj  # A SlurmFile instance must be provided
        self.symm_prec = settings.SYMM_PREC
        self.disp_mag = disp_mag
        self.unstable_threshold = unstable_threshold

        self.__smodes_processor = None
        self.__perturbations = []

        # Set logger for slurm_obj
        self.slurm_obj.set_logger(self._logger)

    def run_program(self):
        ascii_str1 = """
 ____                                _        _       
/ ___| _   _ _ __ ___  _ __ ___  ___| |_ __ _| |_ ___ 
\___ \| | | | '_ ` _ \| '_ ` _ \/ __| __/ _` | __/ _ \ 
 ___) | |_| | | | | | | | | | | \__ \ || (_| | ||  __/
|____/ \__, |_| |_| |_|_| |_| |_|___/\__\__,_|\__\___|
       |___/                                          
 _____                              ____                                      
| ____|_ __   ___ _ __ __ _ _   _  |  _ \ _ __ ___   __ _ _ __ __ _ _ __ ___  
|  _| | '_ \ / _ \ '__/ _` | | | | | |_) | '__/ _ \ / _` | '__/ _` | '_ ` _ \ 
| |___| | | |  __/ | | (_| | |_| | |  __/| | | (_) | (_| | | | (_| | | | | | |
|_____|_| |_|\___|_|  \__, |\__, | |_|   |_|  \___/ \__, |_|  \__,_|_| |_| |_|
                      |___/ |___/                   |___/       
"""
        self._logger.info(ascii_str1)

        # Initialize SmodesProcessor using the provided slurm_obj and the SMODES path from settings.
        smodes_proc = SmodesProcessor(
            abi_file=self.abi_file,
            smodes_input=self.smodes_input,
            target_irrep=self.target_irrep,
            slurm_obj=self.slurm_obj,
            symm_prec=self.symm_prec,
            disp_mag=self.disp_mag,
            unstable_threshold=self.unstable_threshold,
        )
        self.__smodes_processor = smodes_proc
        normalized_phonon_vecs = smodes_proc.symmadapt()

        self._logger.info(f"Phonon Displacement Vectors:\n{smodes_proc.phonon_vecs}")
        self._logger.info(f"Force Constant Evaluations:\n{smodes_proc.fc_evals}")
        self._logger.info(f"Dynamical Frequencies:\n{smodes_proc.dyn_freqs}")
        self._logger.info(f"Normalized Unstable Phonons:\n{normalized_phonon_vecs}")

        if len(normalized_phonon_vecs) == 0:
            self._logger.info("No unstable phonons were found")
        else:
            ascii_str3 = """
  ____      _            _       _   _             
 / ___|__ _| | ___ _   _| | __ _| |_(_)_ __   __ _ 
| |   / _` | |/ __| | | | |/ _` | __| | '_ \ / _` |
| |__| (_| | | (__| |_| | | (_| | |_| | | | | (_| |
 \____\__,_|_|\___|\__,_|_|\__,_|\__|_|_| |_|\__, |
                                             |___/ 
 _____                      _           
| ____|_ __   ___ _ __ __ _(_) ___  ___ 
|  _| | '_ \ / _ \ '__/ _` | |/ _ \/ __|
| |___| | | |  __/ | | (_| | |  __/\__ \ 
|_____|_| |_|\___|_|  \__, |_|\___||___/
                      |___/             
"""
            self._logger.info(ascii_str3)

            # Update smodes processor abinit file:
            smodes_proc.abinit_file.update_unit_cell_parameters()

            # For each unstable phonon, create a new Perturbations instance using the same slurm_obj.
            for i, pert in enumerate(normalized_phonon_vecs):
                pert_obj = Perturbations(
                    name=self.name,
                    num_datapoints=self.num_datapoints,
                    abi_file=smodes_proc.abinit_file,
                    min_amp=self.min_amp,
                    max_amp=self.max_amp,
                    perturbation=pert,
                    slurm_obj=self.slurm_obj,
                )
                self.__perturbations.append(pert_obj)
                pert_obj.generate_perturbations()
                self._logger.info("Generated Perturbations!")
                pert_obj.calculate_energy_of_perturbations()
                self._logger.info(
                    f"Amplitudes of Unstable Phonon {i}: {pert_obj.list_amps}"
                )
                self._logger.info(
                    f"Energies of Unstable Phonon {i}: {pert_obj.results['energies']}"
                )
                pert_obj.record_data("recorded_energies.txt")

        ascii_str4 = """
 _____ _       _     _              _ 
|  ___(_)_ __ (_)___| |__   ___  __| |
| |_  | | '_ \| / __| '_ \ / _ \/ _` |
|  _| | | | | | \__ \ | | |  __/ (_| |
|_|   |_|_| |_|_|___/_| |_|\___|\__,_|
"""

        self._logger.info(ascii_str4)

    def get_smodes_processor(self):
        return self.__smodes_processor

    def get_perturbations(self):
        return self.__perturbations
