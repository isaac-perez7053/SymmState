from symmstate.flpz import SmodesProcessor, FlpzCore, Perturbations
from symmstate.abinit import *


class ElectroTensorProgram(FlpzCore):
    """
    Energy subclass inheriting from flpz.
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
        smodes_path="/home/iperez/isobyu/smodes",
        host_spec='mpirun -hosts=localhost -np 30',
        batch_script_header_file=None,
        symm_prec='0.00001',
        disp_mag='0.001',
        flexo_calculation=False,
        plot_piezo_relaxed=False
    ):
        # Correctly initialize superclass
        super().__init__(name=name, num_datapoints=num_datapoints, abi_file=abi_file, min_amp=min_amp, max_amp=max_amp)

        self.__smodes_processor = None
        self.__perturbations = []
        self.smodes_path = smodes_path
        self.smodes_input = smodes_input
        self.target_irrep = target_irrep
        self.host_spec = host_spec
        self.batch_script_header_file = batch_script_header_file
        self.symm_prec = symm_prec
        self.disp_mag = disp_mag
        self.flexo_calculation = flexo_calculation
        self.plot_piezo_relaxed = plot_piezo_relaxed

    def run_program(self):
        # Ensure you're accessing inherited attributes correctly
        print("Energy program running")
        # This `genstruc` should be initialized in `flpz`
        smodes_file = SmodesProcessor(
            abi_file=self.abi_file,
            smodes_input=self.smodes_input,
            target_irrep=self.target_irrep,
            smodes_path=self.smodes_path,
            host_spec=self.host_spec,
            symm_prec=self.symm_prec,
            disp_mag=self.disp_mag,
            b_script_header_file=self.batch_script_header_file,
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
                    name=self.name,
                    num_datapoints=self.num_datapoints,
                    abinit_file=self.abi_file,
                    min_amp=self.min_amp,
                    max_amp=self.max_amp,
                    perturbation=pert,
                    batch_script_header_file=self.batch_script_header_file,
                    host_spec=self.host_spec
                )

                print(f"Perturbation object successfully initialized")
                self.__perturbations.append(perturbations)
                perturbations.generate_perturbations()
                if self.flexo_calculation:
                    perturbations.calculate_flexo_of_perturbations()
                    perturbations.data_analysis(save_plot=True, flexo=True)
                else: 
                    perturbations.calculate_piezo_of_perturbations()
                    perturbations.data_analysis(save_plot=True, piezo=True, plot_piezo_relaxed_tensor=self.plot_piezo_relaxed)

    def get_smodes_processor(self):
        return self.__smodes_processor

    def get_perturbations(self):
        return self.__perturbations


