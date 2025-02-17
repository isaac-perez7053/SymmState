from symmstate.flpz.smodes_processor import SmodesProcessor
from symmstate.flpz.perturbations import Perturbations
from symmstate.flpz import FlpzCore

class EnergyProgram(FlpzCore):
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
        symm_prec=0.00001,
        disp_mag=0.001,
        unstable_threshold=-20, 
        piezo_calculation=False
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
        self.unstable_threshold = unstable_threshold
        self.piezo_calculation=piezo_calculation

    def run_program(self):
        # Ensure you're accessing inherited attributes correctly
        ascii_string_1 = """
 ____                                _        _       
/ ___| _   _ _ __ ___  _ __ ___  ___| |_ __ _| |_ ___ 
\___ \| | | | '_ ` _ \| '_ ` _ \/ __| __/ _` | __/ _ \ 
 ___) | |_| | | | | | | | | | | \__ \ || (_| | ||  __/
|____/ \__, |_| |_| |_|_| |_| |_|___/\__\__,_|\__\___|
       |___/                                          
 _____ _           _           _____                         
| ____| | ___  ___| |_ _ __ __|_   _|__ _ __  ___  ___  _ __ 
|  _| | |/ _ \/ __| __| '__/ _ \| |/ _ \ '_ \/ __|/ _ \| '__|
| |___| |  __/ (__| |_| | | (_) | |  __/ | | \__ \ (_) | |   
|_____|_|\___|\___|\__|_|  \___/|_|\___|_| |_|___/\___/|_|   
                                                             
 ____                                      
|  _ \ _ __ ___   __ _ _ __ __ _ _ __ ___  
| |_) | '__/ _ \ / _` | '__/ _` | '_ ` _ \ 
|  __/| | | (_) | (_| | | | (_| | | | | | |
|_|   |_|  \___/ \__, |_|  \__,_|_| |_| |_|
                 |___/                                                      
"""
        print(f"{ascii_string_1} \n")
        smodes_file = SmodesProcessor(
            abi_file=self.abi_file,
            smodes_input=self.smodes_input,
            target_irrep=self.target_irrep,
            smodes_path=self.smodes_path,
            host_spec=self.host_spec,
            symm_prec=self.symm_prec,
            disp_mag=self.disp_mag,
            b_script_header_file=self.batch_script_header_file,
            unstable_threshold=self.unstable_threshold,
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
            ascii_string_3 = """
  ____      _            _       _   _             
 / ___|__ _| | ___ _   _| | __ _| |_(_)_ __   __ _ 
| |   / _` | |/ __| | | | |/ _` | __| | '_ \ / _` |
| |__| (_| | | (__| |_| | | (_| | |_| | | | | (_| |
 \____\__,_|_|\___|\__,_|_|\__,_|\__|_|_| |_|\__, |
                                             |___/ 
 _____ _                     _           _        _      
|  ___| | _____  _____   ___| | ___  ___| |_ _ __(_) ___ 
| |_  | |/ _ \ \/ / _ \ / _ \ |/ _ \/ __| __| '__| |/ __|
|  _| | |  __/>  < (_) |  __/ |  __/ (__| |_| |  | | (__ 
|_|   |_|\___/_/\_\___/ \___|_|\___|\___|\__|_|  |_|\___|
                                                         
 _____                             
|_   _|__ _ __  ___  ___  _ __ ___ 
  | |/ _ \ '_ \/ __|/ _ \| '__/ __|
  | |  __/ | | \__ \ (_) | |  \__ \
  |_|\___|_| |_|___/\___/|_|  |___/
                                   
"""
            print(f"{ascii_string_3} \n")

            for i, pert in enumerate(normalized_phonon_vecs):
                perturbations = Perturbations(
                    name=self.name,
                    num_datapoints=self.num_datapoints,
                    abi_file=self.abi_file,
                    min_amp=self.min_amp,
                    max_amp=self.max_amp,
                    perturbation=pert,
                    batch_script_header_file=self.batch_script_header_file,
                    host_spec=self.host_spec
                )

                self.__perturbations.append(perturbations)
                perturbations.generate_perturbations()
                
                # Check whether or not to run just piezoelectric calculation    
                if self.piezo_calculation:
                    perturbations.calculate_piezo_of_perturbations()
                else:
                    perturbations.calculate_flexo_of_perturbations()
                    perturbations.data_analysis(save_plot=True, filename=f"flexo_vs_amplitude_{i}", flexo=True)
                    print("\n")
                    print(f"Flexoelectric tensors of unstable Phonon {i} \n {perturbations.list_flexo_tensors} \n")


                # Printing relevant information
                print(f"Amplitudes of Unstable Phonon {i}: {perturbations.list_amps} \n")
                print(f"Energies of Unstable Phonon {i}: {perturbations.list_energies} \n")
                print(f"Piezoelectric tensors of unstable, \n")
                print(f"Printing clamped tensors,  {perturbations.list_piezo_tensors_clamped} \n")
                print(f"Printing relaxed tensors, {perturbations.list_piezo_tensors_relaxed} \n")
                perturbations.data_analysis(save_plot=True, filename=f"piezo_relaxed_vs_amplitude_{i}", piezo=True, plot_piezo_relaxed_tensor=True)
                perturbations.data_analysis(save_plot=True, filename=f"energy_vs_amplitude_{i}")

        ascii_string_4 = """
 _____ _       _     _              _ 
|  ___(_)_ __ (_)___| |__   ___  __| |
| |_  | | '_ \| / __| '_ \ / _ \/ _` |
|  _| | | | | | \__ \ | | |  __/ (_| |
|_|   |_|_| |_|_|___/_| |_|\___|\__,_|
"""
        print(f"{ascii_string_4} \n")

    def get_smodes_processor(self):
        return self.__smodes_processor

    def get_perturbations(self):
        return self.__perturbations

