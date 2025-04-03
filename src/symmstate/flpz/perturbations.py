import numpy as np
import os
import matplotlib.pyplot as plt
from symmstate.abinit import AbinitFile
from symmstate.flpz import FlpzCore

# This class is going to become a subclass of the programs class
class Perturbations(FlpzCore):
    """
    A class that facilitates the generation and analysis of perturbations in
    an Abinit unit cell, enabling the calculation of energy, piezoelectric,
    and flexoelectric properties.

    Attributes:
        abi_file (str): Path to the Abinit file.
        min_amp (float): Minimum amplitude of perturbations.
        max_amp (float): Maximum amplitude of perturbations.
        num_datapoints (int): Number of datapoints (perturbed cells) to generate.
        pert (np.ndarray): Numpy array representing the perturbations.
        batch_script_header_file (str): Path to the batch script header file.
        host_spec (str): Host specification for running Abinit jobs.
        list_abi_files (list): List of Abinit input file names for perturbations.
        perturbed_objects (list): List of perturbed objects generated.
        list_energies (list): Energies corresponding to each perturbation.
        list_amps (list): Amplitude values for each perturbation step.
        list_flexo_tensors (list): Flexoelectric tensors for each perturbation.
        list_piezo_tensors_clamped (list): Clamped piezoelectric tensors.
        list_piezo_tensors_relaxed (list): Relaxed piezoelectric tensors.
    """

    def __init__(
        self,
        name=None,
        num_datapoints=None,
        abi_file=None,
        min_amp=0,
        max_amp=0.5,
        perturbation=None,
        batch_script_header_file="slurm_file.sh",
        host_spec='mpirun -hosts=localhost -np 30'
    ):
        """
        Initializes the Perturbations instance with additional parameters.

        Args:
            abi_file (str): Path to the Abinit file.
            num_datapoints (int): Number of perturbed unit cells to generate.
            min_amp (float): Minimum amplitude of perturbations.
            max_amp (float): Maximum amplitude of perturbations.
            perturbation (np.ndarray): Numpy array representing the perturbations.
            batch_script_header_file (str): Path to the batch script header file.
            host_spec (str): Host specification for running Abinit jobs.
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

        # Create an AbinitFile instance using the revised constructor.
        self.abinit_file = AbinitFile(
            abi_file=abi_file, batch_script_header_file=batch_script_header_file
        )

        self.pert = np.array(perturbation, dtype=np.float64)
        self.host_spec = str(host_spec)

        self.list_abi_files = []
        self.perturbed_objects = []
        self.list_energies = []
        self.list_amps = []
        self.list_flexo_tensors = []
        self.list_piezo_tensors_clamped = []
        self.list_piezo_tensors_relaxed = []

    def generate_perturbations(self):
        """
        Generates perturbed unit cells based on the given number of datapoints.
        
        Returns:
            list: A list containing the perturbed unit cells.
        """
        # Calculate the step size.
        step_size = (self.max_amp - self.min_amp) / (self.num_datapoints - 1)
        for i in range(self.num_datapoints):
            # Calculate the current amplitude factor.
            current_amp = self.min_amp + i * step_size
            self.list_amps.append(current_amp)
            # Compute the perturbations using the updated perturbations method (inherited from AbinitUnitCell).
            perturbed_values = current_amp * self.pert
            perturbation_result = self.abinit_file.perturbations(
                perturbed_values, coords_are_cartesian=True
            )
            self.perturbed_objects.append(perturbation_result)
        return self.perturbed_objects

    def calculate_energy_of_perturbations(self):
        """
        Runs an energy calculation for each of the Abinit perturbation objects.
        """
        for i, perturbation_object in enumerate(self.perturbed_objects):
            # Update file name using the new unique filename method from AbinitFile.
            perturbation_object.file_name = AbinitFile._get_unique_filename(
                f"{perturbation_object.file_name}_{i}"
            )
            perturbation_object.file_name = os.path.basename(perturbation_object.file_name)
            # Run energy calculation.
            perturbation_object.run_energy_calculation(host_spec=self.host_spec)
            self.list_abi_files.append(f"{perturbation_object.file_name}.abi")
            # Track the job.
            self.abinit_file.running_jobs.append(perturbation_object.running_jobs[-1])
        # Wait for jobs to finish.
        self.abinit_file.wait_for_jobs_to_finish(check_time=90)
        # Extract energy from each perturbed object.
        for obj in self.perturbed_objects:
            obj.grab_energy(f"{obj.file_name}_energy.abo")
            self.list_energies.append(obj.energy)

    def calculate_piezo_of_perturbations(self):
        # Step 1: Run piezo calculations for each perturbed object.
        for i, p_obj in enumerate(self.perturbed_objects):
            p_obj.file_name = AbinitFile._get_unique_filename(f"{p_obj.file_name}_{i}")
            p_obj.file_name = os.path.basename(p_obj.file_name)
            p_obj.run_piezo_calculation(host_spec=self.host_spec)
            self.list_abi_files.append(f"{p_obj.file_name}.abi")
            self.abinit_file.running_jobs.append(p_obj.running_jobs[-1])
        self.abinit_file.wait_for_jobs_to_finish(check_time=300)
        
        # Step 2: Run mrgddb for each perturbed object.
        mrgddb_output_files = []
        for p_obj in self.perturbed_objects:
            output_file = f"{p_obj.file_name}_mrgddb_output"
            content = f"""{output_file}
    Piezo electric calculation of the file {p_obj.file_name} 
    2
    {p_obj.file_name}_piezo_gen_output_DS1_DDB
    {p_obj.file_name}_piezo_gen_output_DS4_DDB
    """
            p_obj.run_mrgddb_file(content)
            mrgddb_output_files.append(output_file)
        self.abinit_file.wait_for_jobs_to_finish(check_time=60, check_once=True)
        
        # Step 3: Run anaddb for each perturbed object.
        anaddb_piezo_files = []
        for i, p_obj in enumerate(self.perturbed_objects):
            anaddb_file_name = p_obj.run_anaddb_file(mrgddb_output_files[i], piezo=True)
            anaddb_piezo_files.append(anaddb_file_name)
        self.abinit_file.wait_for_jobs_to_finish(check_time=60, check_once=True)
        
        # Step 4: Clear the energy list and then collect data.
        self.list_energies = []
        self.list_piezo_tensors_clamped = []
        self.list_piezo_tensors_relaxed = []
        for i, p_obj in enumerate(self.perturbed_objects):
            p_obj.grab_energy(f"{p_obj.file_name}_{i}_piezo.abo")
            p_obj.grab_piezo_tensor(anaddb_file=anaddb_piezo_files[i])
            self.list_energies.append(p_obj.energy)
            self.list_piezo_tensors_clamped.append(p_obj.piezo_tensor_clamped)
            self.list_piezo_tensors_relaxed.append(p_obj.piezo_tensor_relaxed)

    def calculate_flexo_of_perturbations(self):
        """
        Runs an energy, piezoelectric, and flexoelectric calculation for each of the Abinit perturbation objects.
        """
        for i, perturbation_object in enumerate(self.perturbed_objects):
            perturbation_object.file_name = AbinitFile._get_unique_filename(
                f"{perturbation_object.file_name}_{i}"
            )
            perturbation_object.file_name = os.path.basename(perturbation_object.file_name)
            perturbation_object.run_flexo_calculation(host_spec=self.host_spec)
            self.list_abi_files.append(f"{perturbation_object.file_name}.abi")
            self.abinit_file.running_jobs.append(perturbation_object.running_jobs[-1])
        self.abinit_file.wait_for_jobs_to_finish(check_time=600)
        mrgddb_output_files = []
        for p_obj in self.perturbed_objects:
            output_file = f"{p_obj.file_name}_mrgddb_output"
            content = f"""{output_file}
Flexo electric calculation of the file {p_obj.file_name} 
3
{p_obj.file_name}_flexo_gen_output_DS1_DDB
{p_obj.file_name}_flexo_gen_output_DS4_DDB
{p_obj.file_name}_flexo_gen_output_DS5_DDB
"""
            p_obj.run_mrgddb_file(content)
            mrgddb_output_files.append(output_file)
        self.abinit_file.wait_for_jobs_to_finish(check_time=60, check_once=True)
        anaddb_flexo_files = []
        for i, p_obj in enumerate(self.perturbed_objects):
            anaddb_file_name = p_obj.run_anaddb_file(mrgddb_output_files[i], flexo=True)
            anaddb_flexo_files.append(anaddb_file_name)
        self.abinit_file.wait_for_jobs_to_finish(check_time=60, check_once=True)
        anaddb_piezo_files = []
        for i, p_obj in enumerate(self.perturbed_objects):
            anaddb_file_name = p_obj.run_anaddb_file(mrgddb_output_files[i], peizo=True)
            anaddb_piezo_files.append(anaddb_file_name)
        self.abinit_file.wait_for_jobs_to_finish(check_time=60, check_once=True)
        for i, p_obj in enumerate(self.perturbed_objects):
            p_obj.grab_energy(f"{p_obj.file_name}_{i}_flexo.abo")
            p_obj.grab_flexo_tensor(anaddb_file=anaddb_flexo_files[i])
            p_obj.grab_piezo_tensor(anaddb_file=anaddb_piezo_files[i])
            self.list_energies.append(p_obj.energy)
            self.list_piezo_tensors_clamped.append(p_obj.piezo_tensor_clamped)
            self.list_piezo_tensors_relaxed.append(p_obj.piezo_tensor_relaxed)
            self.list_flexo_tensors.append(p_obj.flexo_tensor)

    def _clean_lists(self, amps, to_be_cleaned_list):
        cleaned_amps = []
        cleaned_list = []
        if len(amps) != len(to_be_cleaned_list):
            raise ValueError("Arrays must be the same length!")
        for i, tensor in enumerate(to_be_cleaned_list):
            if tensor is not None:
                cleaned_amps.append(amps[i])
                cleaned_list.append(tensor)
        return cleaned_amps, cleaned_list

    def record_data(self, data_file):
        with open(data_file, "w") as f:
            f.write("Data File \n")
            f.write("Printing Basic Cell File Name: \n")
            f.write(f"{self.abi_file} \n \n")
            f.write("Perturbation Associated with Run: \n")
            f.write(f"{self.pert} \n \n")
            f.write(f"List of Amplitudes: {self.list_amps} \n")
            f.write(f"List of Energies: {self.list_energies} \n \n")
            cleaned_amps, cleaned_piezo_tensors_clamped = self._clean_lists(self.list_amps, self.list_piezo_tensors_clamped)
            if cleaned_piezo_tensors_clamped:
                f.write(f"List of Clamped Piezo Amplitudes: {cleaned_amps} \n")
                f.write("List of Clamped Piezo Electric Tensors: \n")
                for tensor in cleaned_piezo_tensors_clamped:
                    f.write(f"{tensor}  \n \n")
            cleaned_amps, cleaned_piezo_tensors_relaxed = self._clean_lists(self.list_amps, self.list_piezo_tensors_relaxed)
            if cleaned_piezo_tensors_relaxed:
                f.write(f"List of Relaxed Piezo Amplitudes: {cleaned_amps} \n")
                f.write("List of Relaxed Piezo Electric Tensors: \n")
                for tensor in cleaned_piezo_tensors_relaxed:
                    f.write(f"{tensor} \n \n")
            cleaned_amps, cleaned_flexo_tensors = self._clean_lists(self.list_amps, self.list_flexo_tensors)
            if cleaned_flexo_tensors:
                f.write(f"List of Flexo Amplitudes: {cleaned_amps} \n")
                f.write("List of Flexo Electric Tensors: \n")
                for tensor in cleaned_flexo_tensors:
                    f.write(f"{tensor} \n \n")

    def data_analysis(
        self,
        piezo=False,
        flexo=False,
        save_plot=False,
        filename="energy_vs_amplitude",
        component_string="all",
        plot_piezo_relaxed_tensor=False,
    ):
        if flexo:
            if len(self.list_amps) != len(self.list_flexo_tensors):
                raise ValueError("Mismatch between amplitude array and flexoelectric tensors.")
            cleaned_amps, cleaned_flexo_tensors = self._clean_lists(self.list_amps, self.list_flexo_tensors)
            num_components = cleaned_flexo_tensors[0].flatten().size
            if component_string == "all":
                selected_indices = list(range(num_components))
            else:
                try:
                    selected_indices = [int(i) - 1 for i in component_string.split()]
                    if any(i < 0 or i >= num_components for i in selected_indices):
                        raise ValueError
                except ValueError:
                    raise ValueError(f"Invalid input in component_string. Enter numbers between 1 and {num_components}.")
            plot_data = np.zeros((len(cleaned_flexo_tensors), len(selected_indices)))
            for idx, tensor in enumerate(cleaned_flexo_tensors):
                flat_tensor = tensor.flatten()
                plot_data[idx, :] = flat_tensor[selected_indices]
            fig, ax = plt.subplots(figsize=(8, 6))
            for i in range(len(selected_indices)):
                ax.plot(
                    cleaned_amps,
                    plot_data[:, i],
                    linestyle=":",
                    marker="o",
                    markersize=8,
                    linewidth=1.5,
                    label=f"μ_{selected_indices[i] + 1}",
                )
            ax.set_xlabel("x (bohrs)", fontsize=14)
            ax.set_ylabel(r"$\mu_{i,j} \left(\frac{nC}{m}\right)$", fontsize=14)
            ax.set_title("Flexoelectric Tensor Components vs. Amplitude of Displacement", fontsize=16)
            ax.set_xlim(0, self.max_amp)
            ax.set_ylim(np.min(plot_data), np.max(plot_data))
            ax.grid(True)
            ax.tick_params(axis="both", which="major", labelsize=14)
            plt.tight_layout(pad=0.5)
            if save_plot:
                plt.savefig(f"{filename}_flexo.png", bbox_inches="tight")
                print(f"Plot saved as {filename}_flexo.png in {os.getcwd()}")
            plt.show()
        elif piezo:
            if len(self.list_amps) != len(self.list_piezo_tensors_clamped):
                raise ValueError("Mismatch between amplitude array and clamped piezoelectric tensors.")
            cleaned_amps, cleaned_list_piezo_tensors_clamped = self._clean_lists(self.list_amps, self.list_piezo_tensors_clamped)
            num_components = cleaned_list_piezo_tensors_clamped[0].flatten().size
            if component_string == "all":
                selected_indices = list(range(num_components))
            else:
                try:
                    selected_indices = [int(i) - 1 for i in component_string.split()]
                    if any(i < 0 or i >= num_components for i in selected_indices):
                        raise ValueError
                except ValueError:
                    raise ValueError(f"Invalid input in component_string. Enter numbers between 1 and {num_components}.")
            plot_data_clamped = np.zeros((len(cleaned_list_piezo_tensors_clamped), len(selected_indices)))
            for idx, tensor in enumerate(cleaned_list_piezo_tensors_clamped):
                flat_tensor = tensor.flatten()
                plot_data_clamped[idx, :] = flat_tensor[selected_indices]
            fig, ax = plt.subplots(figsize=(8, 6))
            for i in range(len(selected_indices)):
                ax.plot(
                    cleaned_amps,
                    plot_data_clamped[:, i],
                    linestyle=":",
                    marker="o",
                    markersize=8,
                    linewidth=1.5,
                    label=f"μ_{selected_indices[i] + 1}",
                )
            ax.set_xlabel("x (bohrs)", fontsize=14)
            ax.set_ylabel(r"$\mu_{i,j} \left(\frac{nC}{m}\right)$", fontsize=14)
            ax.set_title("Piezoelectric Tensor Components (Clamped) vs. Amplitude of Displacement", fontsize=16)
            ax.set_xlim(0, self.max_amp)
            ax.set_ylim(np.min(plot_data_clamped), np.max(plot_data_clamped))
            ax.grid(True)
            ax.tick_params(axis="both", which="major", labelsize=14)
            filename_based = os.path.basename(filename)
            filename = f"{filename_based}_piezo_clamped.png"
            if save_plot:
                plt.savefig(f"{filename}", bbox_inches="tight")
                print(f"Plot saved as {filename} in {os.getcwd()}")
            plt.show()
            if plot_piezo_relaxed_tensor:
                if len(self.list_amps) != len(self.list_piezo_tensors_relaxed):
                    raise ValueError("Length of amplitude array and relaxed piezoelectric tensors must match!")
                cleaned_amps, cleaned_list_piezo_tensors_relaxed = self._clean_lists(self.list_amps, self.list_piezo_tensors_relaxed)
                plot_data_relaxed = np.zeros((len(cleaned_list_piezo_tensors_relaxed), len(selected_indices)))
                for idx, tensor in enumerate(cleaned_list_piezo_tensors_relaxed):
                    flat_tensor = tensor.flatten()
                    plot_data_relaxed[idx, :] = flat_tensor[selected_indices]
                fig, ax = plt.subplots(figsize=(8, 6))
                for i in range(len(selected_indices)):
                    ax.plot(
                        cleaned_amps,
                        plot_data_relaxed[:, i],
                        linestyle=":",
                        marker="o",
                        markersize=8,
                        linewidth=1.5,
                        label=f"μ_{selected_indices[i] + 1}",
                    )
                ax.set_xlabel("x (bohrs)", fontsize=14)
                ax.set_ylabel(r"$\mu_{i,j} \left(\frac{nC}{m}\right)$", fontsize=14)
                ax.set_title("Piezoelectric Tensor Components (Relaxed) vs. Amplitude of Displacement", fontsize=16)
                ax.set_xlim(0, self.max_amp)
                ax.set_ylim(np.min(plot_data_relaxed), np.max(plot_data_relaxed))
                ax.grid(True)
                ax.legend(loc="best", fontsize=12)
                ax.tick_params(axis="both", which="major", labelsize=14)
                plt.tight_layout(pad=0.5)
                filename_based = os.path.basename(filename)
                filename = f"{filename_based}_piezo_relaxed.png"
                if save_plot:
                    plt.savefig(f"{filename}", bbox_inches="tight")
                    print(f"Plot saved as {filename} in {os.getcwd()}")
                plt.show()
        else:
            if len(self.list_energies) != len(self.list_amps):
                raise ValueError("Length of list_energies and list_amps must match.")
            fig, ax = plt.subplots()
            ax.plot(self.list_amps, self.list_energies, marker="o", linestyle="-", color="b")
            ax.set_title("Energy vs Amplitude of Perturbations")
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("Energy")
            x_margin = 0.1 * (self.max_amp - self.min_amp)
            y_margin = 0.1 * (max(self.list_energies) - min(self.list_energies))
            ax.set_xlim(self.min_amp - x_margin, self.max_amp + x_margin)
            ax.set_ylim(min(self.list_energies) - y_margin, max(self.list_energies) + y_margin)
            ax.grid(True)
            plt.tight_layout(pad=0.5)
            if save_plot:
                plt.savefig(filename, bbox_inches="tight")
                print(f"Plot saved as {filename} in {os.getcwd()}")
            plt.show()

