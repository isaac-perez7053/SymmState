import numpy as np
import os
import matplotlib.pyplot as plt
from symmstate.abinit import AbinitFile
from symmstate.flpz import FlpzCore

class Perturbations(FlpzCore):
    """
    A class that facilitates the generation and analysis of perturbations in
    an Abinit unit cell, enabling the calculation of energy, piezoelectric,
    and flexoelectric properties.

    Attributes:
        abi_file (str): Path to the Abinit file.
        min_amp (float): Minimum amplitude of perturbations.
        max_amp (float): Maximum amplitude of perturbations.
        num_datapoints (int): Number of perturbed cells to generate.
        pert (np.ndarray): Array representing the base perturbation.
        slurm_obj (SlurmFile): An instance of SlurmFile to handle job submission.
        list_abi_files (list): List of generated Abinit input filenames.
        perturbed_objects (list): List of perturbed AbinitFile objects.
        list_amps (list): Amplitude values for each perturbation step.
        results (dict): Dictionary storing extracted outputs from calculations.
                         Expected keys include:
                           - 'energies': list of energy values
                           - 'flexo': list of flexoelectric tensors
                           - 'piezo': a sub-dictionary with keys 'clamped' and 'relaxed'
    """

    def __init__(
        self,
        name=None,
        num_datapoints=None,
        abi_file=None,
        min_amp=0,
        max_amp=0.5,
        perturbation=None,
        slurm_obj=None
    ):
        """
        Initializes the Perturbations instance with additional parameters.

        Args:
            abi_file (str): Path to the Abinit file.
            num_datapoints (int): Number of perturbed unit cells to generate.
            min_amp (float): Minimum amplitude of perturbations.
            max_amp (float): Maximum amplitude of perturbations.
            perturbation (np.ndarray): Array representing the base perturbation.
            slurm_obj (SlurmFile): An instance of SlurmFile to handle job submission.
        """
        if not isinstance(perturbation, np.ndarray):
            raise ValueError("perturbation should be a numpy array.")

        # Store key parameters for later use.
        self.abi_file = abi_file
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.num_datapoints = num_datapoints

        # Initialize the base class.
        super().__init__(
            name=name,
            num_datapoints=num_datapoints,
            abi_file=abi_file,
            min_amp=min_amp,
            max_amp=max_amp,
        )

        # Create an AbinitFile instance using the provided slurm_obj.
        self.abinit_file = AbinitFile(
            abi_file=abi_file, slurm_obj=slurm_obj
        )

        self.pert = np.array(perturbation, dtype=np.float64)

        self.list_abi_files = []
        self.perturbed_objects = []
        self.list_amps = []

        # Use a results dictionary to store extracted data.
        self.results = {
            "energies": [],
            "flexo": [],
            "piezo": {
                "clamped": [],
                "relaxed": []
            }
        }

    def generate_perturbations(self):
        """
        Generates perturbed unit cells based on the given number of datapoints.
        Returns:
            list: A list of perturbed AbinitFile objects.
        """
        # Calculate the step size.
        step_size = (self.max_amp - self.min_amp) / (self.num_datapoints - 1)
        for i in range(self.num_datapoints):
            current_amp = self.min_amp + i * step_size
            self.list_amps.append(current_amp)
            # Compute perturbed values and obtain a new AbinitFile object.
            perturbed_values = current_amp * self.pert
            perturbation_result = self.abinit_file.perturbations(
                perturbed_values, coords_are_cartesian=True
            )
            self.perturbed_objects.append(perturbation_result)
        return self.perturbed_objects

    def calculate_energy_of_perturbations(self):
        """
        Runs an energy calculation for each perturbed AbinitFile object and stores
        the energy in self.results["energies"].
        """
        for i, obj in enumerate(self.perturbed_objects):
            # Update filename to ensure uniqueness.
            obj.file_name = AbinitFile._get_unique_filename(f"{obj.file_name}_{i}")
            obj.file_name = os.path.basename(obj.file_name)
            obj.run_energy_calculation()  # Use slurm_obj from AbinitFile
            self.list_abi_files.append(f"{obj.file_name}.abi")
            # Append energy result.
            obj.grab_energy(f"{obj.file_name}_energy.abo")
            self.results["energies"].append(obj.energy)
        self.abinit_file.wait_for_jobs_to_finish(check_time=90)

    def calculate_piezo_of_perturbations(self):
        """
        Runs piezoelectric calculations for each perturbed object and stores the
        energies and piezoelectric tensors (both clamped and relaxed) in self.results.
        """
        for i, obj in enumerate(self.perturbed_objects):
            obj.file_name = AbinitFile._get_unique_filename(f"{obj.file_name}_{i}")
            obj.file_name = os.path.basename(obj.file_name)
            obj.run_piezo_calculation()
            self.list_abi_files.append(f"{obj.file_name}.abi")
            # Append job id using the slurm_obj from the AbinitFile.
            self.abinit_file.slurm_obj.running_jobs.append(obj.slurm_obj.running_jobs[-1])
        self.abinit_file.slurm_obj.wait_for_jobs_to_finish(check_time=300)
        
        mrgddb_output_files = []
        for obj in self.perturbed_objects:
            output_file = f"{obj.file_name}_mrgddb_output"
            content = (f"{output_file}\n"
                    f"Piezoelectric calculation of file {obj.file_name}\n"
                    "2\n"
                    f"{obj.file_name}_piezo_gen_output_DS1_DDB\n"
                    f"{obj.file_name}_piezo_gen_output_DS4_DDB\n")
            obj.run_mrgddb_file(content)
            mrgddb_output_files.append(output_file)
        self.abinit_file.slurm_obj.wait_for_jobs_to_finish(check_time=60, check_once=True)
        
        anaddb_piezo_files = []
        for i, obj in enumerate(self.perturbed_objects):
            anaddb_file_name = obj.run_anaddb_file(mrgddb_output_files[i], piezo=True)
            anaddb_piezo_files.append(anaddb_file_name)
        self.abinit_file.slurm_obj.wait_for_jobs_to_finish(check_time=60, check_once=True)
        
        # Clear any previous energy and piezo results.
        self.results["energies"] = []
        self.results["piezo"]["clamped"] = []
        self.results["piezo"]["relaxed"] = []
        for i, obj in enumerate(self.perturbed_objects):
            obj.grab_energy(f"{obj.file_name}_{i}_piezo.abo")
            obj.grab_piezo_tensor(anaddb_file=anaddb_piezo_files[i])
            self.results["energies"].append(obj.energy)
            self.results["piezo"]["clamped"].append(obj.piezo_tensor_clamped)
            self.results["piezo"]["relaxed"].append(obj.piezo_tensor_relaxed)

    def calculate_flexo_of_perturbations(self):
        """
        Runs flexoelectric calculations for each perturbed object and stores the
        energies, flexoelectric tensors, and piezoelectric tensors in self.results.
        """
        for i, obj in enumerate(self.perturbed_objects):
            obj.file_name = AbinitFile._get_unique_filename(f"{obj.file_name}_{i}")
            obj.file_name = os.path.basename(obj.file_name)
            obj.run_flexo_calculation()
            self.list_abi_files.append(f"{obj.file_name}.abi")
            # Use the slurm_obj for job tracking.
            self.abinit_file.slurm_obj.running_jobs.append(obj.slurm_obj.running_jobs[-1])
        self.abinit_file.slurm_obj.wait_for_jobs_to_finish(check_time=600)
        
        mrgddb_output_files = []
        for obj in self.perturbed_objects:
            output_file = f"{obj.file_name}_mrgddb_output"
            content = (f"{output_file}\n"
                    f"Flexoelectric calculation of file {obj.file_name}\n"
                    "3\n"
                    f"{obj.file_name}_flexo_gen_output_DS1_DDB\n"
                    f"{obj.file_name}_flexo_gen_output_DS4_DDB\n"
                    f"{obj.file_name}_flexo_gen_output_DS5_DDB\n")
            obj.run_mrgddb_file(content)
            mrgddb_output_files.append(output_file)
        self.abinit_file.slurm_obj.wait_for_jobs_to_finish(check_time=60, check_once=True)
        
        anaddb_flexo_files = []
        for i, obj in enumerate(self.perturbed_objects):
            anaddb_file_name = obj.run_anaddb_file(mrgddb_output_files[i], flexo=True)
            anaddb_flexo_files.append(anaddb_file_name)
        self.abinit_file.slurm_obj.wait_for_jobs_to_finish(check_time=60, check_once=True)
        
        anaddb_piezo_files = []
        for i, obj in enumerate(self.perturbed_objects):
            anaddb_file_name = obj.run_anaddb_file(mrgddb_output_files[i], peizo=True)
            anaddb_piezo_files.append(anaddb_file_name)
        self.abinit_file.slurm_obj.wait_for_jobs_to_finish(check_time=60, check_once=True)
        
        for i, obj in enumerate(self.perturbed_objects):
            obj.grab_energy(f"{obj.file_name}_{i}_flexo.abo")
            obj.grab_flexo_tensor(anaddb_file=anaddb_flexo_files[i])
            obj.grab_piezo_tensor(anaddb_file=anaddb_piezo_files[i])
            self.results["energies"].append(obj.energy)
            self.results["flexo"].append(obj.flexo_tensor)
            self.results["piezo"]["clamped"].append(obj.piezo_tensor_clamped)
            self.results["piezo"]["relaxed"].append(obj.piezo_tensor_relaxed)


    def record_data(self, data_file):
        """
        Writes a summary of the run to a file, including the extracted data from the 
        perturbation calculations stored in self.results.
        """
        with open(data_file, "w") as f:
            f.write("Data File\n")
            f.write("Basic Cell File Name:\n")
            f.write(f"{self.abi_file}\n\n")
            f.write("Perturbation Associated with Run:\n")
            f.write(f"{self.pert}\n\n")
            f.write(f"List of Amplitudes: {self.list_amps}\n")
            f.write("Extracted Results:\n")
            for key, value in self.results.items():
                f.write(f"{key}: {value}\n")

    def data_analysis(
        self,
        piezo=False,
        flexo=False,
        save_plot=False,
        filename="energy_vs_amplitude",
        component_string="all",
        plot_piezo_relaxed_tensor=False,
    ):
        """
        Plots the desired property (energy, piezo, or flexo tensor component)
        versus the amplitude of displacement.
        """
        if flexo:
            if len(self.list_amps) != len(self.results["flexo"]):
                raise ValueError("Mismatch between amplitudes and flexoelectric tensors.")
            cleaned_amps = self.list_amps
            flexo_tensors = self.results["flexo"]
            num_components = flexo_tensors[0].flatten().size
            if component_string == "all":
                selected_indices = list(range(num_components))
            else:
                try:
                    selected_indices = [int(i) - 1 for i in component_string.split()]
                    if any(i < 0 or i >= num_components for i in selected_indices):
                        raise ValueError
                except ValueError:
                    raise ValueError(f"Invalid input in component_string. Enter numbers between 1 and {num_components}.")
            plot_data = np.zeros((len(flexo_tensors), len(selected_indices)))
            for idx, tensor in enumerate(flexo_tensors):
                flat_tensor = tensor.flatten()
                plot_data[idx, :] = flat_tensor[selected_indices]
            fig, ax = plt.subplots(figsize=(8, 6))
            for i in range(len(selected_indices)):
                ax.plot(cleaned_amps, plot_data[:, i], linestyle=":", marker="o",
                        markersize=8, linewidth=1.5, label=f"μ_{selected_indices[i] + 1}")
            ax.set_xlabel("Amplitude (bohrs)", fontsize=14)
            ax.set_ylabel(r"$\mu_{i,j} \left(\frac{nC}{m}\right)$", fontsize=14)
            ax.set_title("Flexoelectric Tensor Components vs. Amplitude", fontsize=16)
            ax.grid(True)
            ax.tick_params(axis="both", which="major", labelsize=14)
            plt.tight_layout(pad=0.5)
            if save_plot:
                plt.savefig(f"{filename}_flexo.png", bbox_inches="tight")
            plt.show()
        elif piezo:
            # Similar plotting logic can be applied for piezo tensors using self.results["piezo"]
            pass
        else:
            if len(self.list_amps) != len(self.results["energies"]):
                raise ValueError("Mismatch between amplitudes and energy values.")
            fig, ax = plt.subplots()
            ax.plot(self.list_amps, self.results["energies"], marker="o", linestyle="-", color="b")
            ax.set_title("Energy vs Amplitude of Perturbations")
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("Energy")
            x_margin = 0.1 * (self.max_amp - self.min_amp)
            y_margin = 0.1 * (max(self.results["energies"]) - min(self.results["energies"]))
            ax.set_xlim(self.min_amp - x_margin, self.max_amp + x_margin)
            ax.set_ylim(min(self.results["energies"]) - y_margin, max(self.results["energies"]) + y_margin)
            ax.grid(True)
            plt.tight_layout(pad=0.5)
            if save_plot:
                plt.savefig(filename, bbox_inches="tight")
            plt.show()


