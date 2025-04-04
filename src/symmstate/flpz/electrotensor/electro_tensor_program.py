from symmstate.config.settings import settings
from symmstate.flpz.smodes_processor import SmodesProcessor
from symmstate.flpz.perturbations import Perturbations
from symmstate.flpz import FlpzCore

class ElectroTensorProgram(FlpzCore):
    """
    ElectroTensorProgram runs a series of SMODES and perturbation calculations to analyze 
    energy, piezoelectric, and flexoelectric properties.
    """

    def __init__(
        self,
        name=None,
        num_datapoints=None,
        abi_file=None,
        min_amp=None,
        max_amp=None,
        smodes_input=None,
        target_irrep=None,
        slurm_obj=None,
        symm_prec=1e-5,
        disp_mag=0.001,
        unstable_threshold=-20, 
        piezo_calculation=False
    ):
        # Correctly initialize superclass.
        super().__init__(name=name, num_datapoints=num_datapoints, abi_file=abi_file, min_amp=min_amp, max_amp=max_amp)
        self._logger.info("Initializing ElectroTensorProgram")

        self.smodes_input = smodes_input
        self.target_irrep = target_irrep
        self.slurm_obj = slurm_obj  # A SlurmFile instance must be provided.
        self.symm_prec = symm_prec
        self.disp_mag = disp_mag
        self.unstable_threshold = unstable_threshold
        self.piezo_calculation = piezo_calculation

        self.__smodes_processor = None
        self.__perturbations = []

    def run_program(self):
        ascii_str1 = (
            "\n ____                                _        _       \n"
            "/ ___| _   _ _ __ ___  _ __ ___  ___| |_ __ _| |_ ___ \n"
            "\\___ \\| | | | '_ ` _ \\| '_ ` _ \\/ __| __/ _` | __/ _ \\ \n"
            " ___) | |_| | | | | | | | | | | \\__ \\ || (_| | ||  __/ \n"
            "|____/ \\__, |_| |_| |_|_| |_| |_|___/\\__\\__,_|\\__\\___| \n"
            "       |___/                                          \n"
            " _____ _           _           _____                         \n"
            "| ____| | ___  ___| |_ _ __ __|_   _|__ _ __  ___  ___  _ __ \n"
            "|  _| | |/ _ \\/ __| __| '__/ _ \\| |/ _ \\ '_ \\/ __|/ _ \\| '__|\n"
            "| |___| |  __/ (__| |_| | | (_) | |  __/ | | \\__ \\ (_) | |   \n"
            "|_____|_|\\___|\\___|\\__|_|  \\___/|_|\\___|_| |_|___/\\___/|_|   \n"
        )
        self._logger.info(ascii_str1)

        # Initialize SmodesProcessor using the provided slurm_obj and the global SMODES path.
        # Note: The SMODES executable path is taken from settings.SMODES_PATH.
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

        if not normalized_phonon_vecs:
            self._logger.info("No unstable phonons were found")
        else:
            ascii_str3 = (
                "\n  ____      _            _       _   _             \n"
                " / ___|__ _| | ___ _   _| | __ _| |_(_)_ __   __ _ \n"
                "| |   / _` | |/ __| | | | |/ _` | __| | '_ \\ / _` |\n"
                "| |__| (_| | | (__| |_| | | (_| | |_| | | | | (_| |\n"
                " \\____\\__,_|_|\\___|\\__,_|_|\\__,_|\\__|_|_| |_|\\__, |\n"
                "                                             |___/ \n"
                " _____ _                     _           _        _      \n"
                "|_   _|__ _ __  ___  ___  _ __ ___ \n"
                "  | |/ _ \\ '_ \\/ __|/ _ \\| '__/ __|\n"
                "  | |  __/ | | \\__ \\ (_) | |  \\__ \\\n"
                "  |_|\___|_| |_|___/\\___/|_|  |___/\n"
            )
            self._logger.info(ascii_str3)

            # Update the Abinit file based on SMODES output.
            self.update_abinit_file("dist_0.abi")

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
                # Run either piezo or flexoelectric calculations based on the flag.
                if self.piezo_calculation:
                    pert_obj.calculate_piezo_of_perturbations()
                else:
                    pert_obj.calculate_flexo_of_perturbations()
                    pert_obj.data_analysis(save_plot=True, filename=f"flexo_vs_amplitude_{i}", flexo=True)
                    self._logger.info(f"Flexoelectric tensors of unstable Phonon {i}:\n{pert_obj.list_flexo_tensors}")
                # Record the data into a file.
                data_file_name = f"data_file_{i}.txt"
                pert_obj.record_data(data_file_name)
                self._logger.info(f"Amplitudes of Unstable Phonon {i}: {pert_obj.list_amps}")
                self._logger.info(f"Energies of Unstable Phonon {i}: {pert_obj.results['energies']}")
                self._logger.info(f"Clamped Piezoelectric tensors of unstable Phonon {i}:\n{pert_obj.list_piezo_tensors_clamped}")
                self._logger.info(f"Relaxed Piezoelectric tensors of unstable Phonon {i}:\n{pert_obj.list_piezo_tensors_relaxed}")
                pert_obj.data_analysis(save_plot=True, filename=f"piezo_relaxed_vs_amplitude_{i}", piezo=True, plot_piezo_relaxed_tensor=True)
                pert_obj.data_analysis(save_plot=True, filename=f"energy_vs_amplitude_{i}")

        ascii_str4 = (
            "\n _____ _       _     _              _ \n"
            "|  ___(_)_ __ (_)___| |__   ___  __| |\n"
            "| |_  | | '_ \\| / __| '_ \\ / _ \\/ _` |\n"
            "|  _| | | | | | \\__ \\ | | |  __/ (_| |\n"
            "|_|   |_|_| |_|_|___/_| |_|\\___|\\__,_|\n"
        )
        self._logger.info(ascii_str4)

    def get_smodes_processor(self):
        return self.__smodes_processor

    def get_perturbations(self):
        return self.__perturbations

