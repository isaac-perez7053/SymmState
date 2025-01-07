from symmstate import FlpzCore
from symmstate.abinit import *


class EnergyProgram(FlpzCore):
    """
    Energy subclass inheriting from flpz.
    """

    def __init__(
        self,
        input_file,
        smodes_input,
        target_irrep,
        smodes_path="/home/iperez/isobyu/smodes",
    ):
        # Correctly initialize superclass
        super().__init__(input_file, smodes_input, target_irrep)
        self.__smodes_processor = None
        self.__perturbations = []
        self.smodes_path = smodes_path

    def run_program(self):
        # Ensure you're accessing inherited attributes correctly
        print("Energy program running")
        # This `genstruc` should be initialized in `flpz`
        smodes_file = SmodesProcessor(
            abi_file=self.genstruc,
            smodes_input=self.smodes_input,
            target_irrep=self.target_irrep,
            b_script_header_file=self.batch_header_file,
            smodes_path=self.smodes_path,
        )
        self.__smodes_processor = smodes_file
        normalized_phonon_vecs = smodes_file.symmadapt()

        print(
            f"Printing Phonon Displacement Vectors: \n \n {smodes_file.phonon_vecs} \n"
        )
        print(f"Printing fc_evals: \n \n {smodes_file.fc_evals} \n")
        print(f"Printing DynFreqs: \n \n {smodes_file.dyn_freqs} \n")

        print(f"normalized unstable phonons: \n \n {normalized_phonon_vecs} \n")
        if len(normalized_phonon_vecs) == 0:
            print("No unstable phonons were found")
        else:
            for pert in normalized_phonon_vecs:
                perturbations = Perturbations(
                    abinit_file=self.genstruc,
                    min_amp=self.min_amp,
                    max_amp=self.max_amp,
                    perturbation=pert,
                    batch_script_header_file=self.batch_header_file,
                )

                print(f"Perturbation object successfully initialized")
                self.__perturbations.append(perturbations)
                perturbations.generate_perturbations(num_datapoints=self.num_datapoints)
                perturbations.calculate_energy_of_perturbations()
                perturbations.data_analysis(save_plot=True)

    def get_smodes_processor(self):
        return self.__smodes_processor

    def get_perturbations(self):
        return self.__perturbations


# TODO: If the user decides to run the program through the terminal \
if __name__ == "__main__":
    pass
