import numpy as np
import os
import matplotlib.pyplot as plt
from symmstate.abinit import AbinitFile
from symmstate.flpz import FlpzCore
from symmstate.utils import DataParser
from symmstate.slurm import *
from typing import List


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
        name: str = None,
        num_datapoints: int = None,
        abi_file: AbinitFile = None,
        min_amp: int = 0,
        max_amp: int = 0.5,
        perturbation: np.ndarray = None,
        slurm_obj: SlurmFile = None,
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
        self.abinit_file = abi_file
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.num_datapoints = num_datapoints

        # Initialize the base class.
        super().__init__(
            name=name,
            num_datapoints=num_datapoints,
            abi_file=f"{self.abinit_file.file_name}.abi",
            min_amp=min_amp,
            max_amp=max_amp,
        )

        self.pert = np.array(perturbation, dtype=np.float64)

        self.list_abi_files = []
        self.perturbed_objects: List[AbinitFile] = []
        self.list_amps = []

        self.slurm_obj = slurm_obj

        # Use a results dictionary to store extracted data.
        self.results = {
            "energies": [],
            "flexo": [],
            "piezo": {"clamped": [], "relaxed": []},
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
        for i, obj in enumerate(self.perturbed_objects):
            obj.file_name = AbinitFile._get_unique_filename(f"{obj.file_name}_{i}")
            obj.file_name = os.path.basename(obj.file_name)
            # Now explicitly pass the slurm_obj
            output_file = obj.run_energy_calculation(slurm_obj=self.slurm_obj)
            self.list_abi_files.append(output_file)

        self.slurm_obj.wait_for_jobs_to_finish(check_time=90)
        # Append just one energy from the final ABO or whichever logic your code uses

        for i, obj in enumerate(self.perturbed_objects):
            energy = DataParser.grab_energy(
                f"{obj.file_name}_energy.abo", logger=self._logger
            )
            self.results["energies"].append(energy)

    def calculate_piezo_of_perturbations(self):
        """
        Runs piezoelectric calculations for each perturbed object and stores the
        energies and piezoelectric tensors (both clamped and relaxed) in self.results.
        """
        for i, obj in enumerate(self.perturbed_objects):
            obj.file_name = AbinitFile._get_unique_filename(f"{obj.file_name}_{i}")
            obj.file_name = os.path.basename(obj.file_name)
            output_file = obj.run_piezo_calculation(slurm_obj=self.slurm_obj)
            self.list_abi_files.append(output_file)

        self.slurm_obj.wait_for_jobs_to_finish(check_time=300)

        mrgddb_output_files = []
        for obj in self.perturbed_objects:
            output_file = f"{obj.file_name}_mrgddb_output"
            content = (
                f"{output_file}\n"
                f"Piezoelectric calculation of file {obj.file_name}\n"
                "2\n"
                f"{obj.file_name}_piezo_gen_output_DS1_DDB\n"
                f"{obj.file_name}_piezo_gen_output_DS4_DDB\n"
            )
            obj.run_mrgddb(content=content)
            mrgddb_output_files.append(output_file)
        self.slurm_obj.wait_for_jobs_to_finish(check_time=60, check_once=True)

        anaddb_piezo_files = []
        for i, obj in enumerate(self.perturbed_objects):
            anaddb_file_name = obj.run_anaddb_file(mrgddb_output_files[i], piezo=True)
            anaddb_piezo_files.append(anaddb_file_name)
        self.slurm_obj.wait_for_jobs_to_finish(check_time=60, check_once=True)

        # Clear any previous energy and piezo results.
        self.results["energies"] = []
        self.results["piezo"]["clamped"] = []
        self.results["piezo"]["relaxed"] = []
        for i, obj in enumerate(self.perturbed_objects):
            energy = DataParser.grab_energy(
                f"{obj.file_name}_{i}_piezo.abo", logger=self._logger
            )
            clamped_tensor, relaxed_tensor = DataParser.grab_piezo_tensor(
                anaddb_file=anaddb_piezo_files[i], logger=self._logger
            )
            self.results["energies"].append(energy)
            self.results["piezo"]["clamped"].append(clamped_tensor)
            self.results["piezo"]["relaxed"].append(relaxed_tensor)

    def calculate_flexo_of_perturbations(self):
        """
        Runs flexoelectric calculations for each perturbed object and stores the
        energies, flexoelectric tensors, and piezoelectric tensors in self.results.
        """

        # Run the flexoelectric calculations for each of the datapoints
        for i, obj in enumerate(self.perturbed_objects):
            obj.file_name = AbinitFile._get_unique_filename(f"{obj.file_name}_{i}")
            obj.file_name = os.path.basename(obj.file_name)
            output_file = obj.run_flexo_calculation(slurm_obj=self.slurm_obj)
            self.list_abi_files.append(output_file)

        self.slurm_obj.wait_for_jobs_to_finish(check_time=600)

        # Merge all of the datasets for each datapoint
        mrgddb_output_files = []
        for obj in self.perturbed_objects:
            output_file = f"{obj.file_name}_mrgddb_output"
            content = (
                f"{output_file}\n"
                f"Flexoelectric calculation of file {obj.file_name}\n"
                "3\n"
                f"{obj.file_name}_flexo_gen_output_DS1_DDB\n"
                f"{obj.file_name}_flexo_gen_output_DS4_DDB\n"
                f"{obj.file_name}_flexo_gen_output_DS5_DDB\n"
            )
            _ = obj.run_mrgddb(content)
            mrgddb_output_files.append(output_file)
        self.slurm_obj.wait_for_jobs_to_finish(check_time=60, check_once=True)

        # Run anaddb analysis for each of the flexoelectric calculations
        anaddb_flexo_files = []
        for i, obj in enumerate(self.perturbed_objects):
            anaddb_file_name = obj.run_anaddb_file(mrgddb_output_files[i], flexo=True)
            anaddb_flexo_files.append(anaddb_file_name)
        self.slurm_obj.wait_for_jobs_to_finish(check_time=60, check_once=True)

        # Run anaddb analysis for each of the piezoelectric datapoints
        anaddb_piezo_files = []
        for i, obj in enumerate(self.perturbed_objects):
            anaddb_file_name = obj.run_anaddb_file(mrgddb_output_files[i], peizo=True)
            anaddb_piezo_files.append(anaddb_file_name)
        self.slurm_obj.wait_for_jobs_to_finish(check_time=60, check_once=True)

        # Grab the desired information from the output files
        for i, obj in enumerate(self.perturbed_objects):
            energy = DataParser.grab_energy(
                f"{obj.file_name}_{i}_flexo.abo", logger=self._logger
            )
            flexotensor = DataParser.grab_flexo_tensor(
                anaddb_file=anaddb_flexo_files[i], logger=self._logger
            )
            clamped_tensor, relaxed_tensor = DataParser.grab_piezo_tensor(
                anaddb_file=anaddb_piezo_files[i], logger=self._logger
            )
            self.results["energies"].append(energy)
            self.results["flexo"].append(flexotensor)
            self.results["piezo"]["clamped"].append(clamped_tensor)
            self.results["piezo"]["relaxed"].append(relaxed_tensor)

    def record_data(self, data_file):
        """
        Writes a summary of the run to a file in the format expected by data-analysis.py.
        This function retrieves the relevant arrays from self.results, including
        flexo- and piezo-related tensors (clamped & relaxed).

        Keys expected in self.results:
        - "amps" -> list of amplitude values
        - "energies" -> list of energies
        - "flexo_amps" -> list of amplitudes used specifically for flexo data
        - "flexo_tensors" -> list of 9x6 arrays for flexo
        - "clamped_piezo_tensors" -> list of NxM arrays for clamped piezo data
        - "relaxed_piezo_tensors" -> list of NxM arrays for relaxed piezo data
        """
        # Useful when analyzing multiple unstable phonons
        data_file = self._get_unique_filename(base_name=data_file)
        with open(data_file, "w") as f:
            # Basic info
            f.write("Data File\n")
            f.write("Basic Cell File Name:\n")
            f.write(f"{self.abi_file}\n\n")
            f.write("Perturbation Associated with Run:\n")
            f.write(f"{self.pert}\n\n")

            # Extract from self.results
            amps = self.results.get("amps", [])
            energies = self.results.get("energies", [])
            flexo_amps = self.results.get("flexo_amps", [])
            flexo_tensors = self.results.get("flexo_tensors", [])

            # Piezo data (clamped, relaxed)
            clamped_tensors = self.results.get("clamped_piezo_tensors", [])
            relaxed_tensors = self.results.get("relaxed_piezo_tensors", [])

            # Required lines for data-analysis.py
            f.write(f"List of Amplitudes: {amps}\n")
            f.write(f"List of Energies: {energies}\n")
            f.write(f"List of Flexo Amplitudes: {flexo_amps}\n")

            # 1) Flexo Tensors
            f.write("List of Flexo Electric Tensors:\n")
            for tensor in flexo_tensors:
                for row in tensor:
                    row_str = " ".join(str(x) for x in row)
                    f.write(f"[ {row_str} ]\n")

            # 2) Clamped Piezo Tensors
            f.write("List of Clamped Piezo Tensors:\n")
            for tensor in clamped_tensors:
                for row in tensor:
                    row_str = " ".join(str(x) for x in row)
                    f.write(f"[ {row_str} ]\n")

            # 3) Relaxed Piezo Tensors
            f.write("List of Relaxed Piezo Tensors:\n")
            for tensor in relaxed_tensors:
                for row in tensor:
                    row_str = " ".join(str(x) for x in row)
                    f.write(f"[ {row_str} ]\n")

            # Any additional fields can be appended here,
            # but maintain the above exact line headers for data-analysis.py to parse them.

    def data_analysis(
        self,
        piezo=False,
        flexo=False,
        save_plot=False,
        filename="energy_vs_amplitude",
        component_string="all",
    ):
        """
        Plots the desired property (energy, piezo, or flexo tensor component)
        versus the amplitude of displacement.
        """
        if flexo:
            if len(self.list_amps) != len(self.results["flexo"]):
                raise ValueError(
                    "Mismatch between amplitudes and flexoelectric tensors."
                )
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
                    raise ValueError(
                        f"Invalid input in component_string. Enter numbers between 1 and {num_components}."
                    )
            plot_data = np.zeros((len(flexo_tensors), len(selected_indices)))
            for idx, tensor in enumerate(flexo_tensors):
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
            ax.plot(
                self.list_amps,
                self.results["energies"],
                marker="o",
                linestyle="-",
                color="b",
            )
            ax.set_title("Energy vs Amplitude of Perturbations")
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("Energy")
            x_margin = 0.1 * (self.max_amp - self.min_amp)
            y_margin = 0.1 * (
                max(self.results["energies"]) - min(self.results["energies"])
            )
            ax.set_xlim(self.min_amp - x_margin, self.max_amp + x_margin)
            ax.set_ylim(
                min(self.results["energies"]) - y_margin,
                max(self.results["energies"]) + y_margin,
            )
            ax.grid(True)
            plt.tight_layout(pad=0.5)
            if save_plot:
                plt.savefig(filename, bbox_inches="tight")
            plt.show()
