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
        min_amp (float): Minimum amplitude of perturbations.
        max_amp (float): Maximum amplitude of perturbations.
        pert (np.ndarray): Numpy array representing the perturbations.
        batchScriptHeader_path (str): Path to the batch script header file.
        list_abi_files (list): List of Abinit input files for perturbations.
        perturbed_objects (list): List of perturbed objects generated.
        list_energies (list): Energies corresponding to each perturbation.
        list_amps (list): Amplitude values for each perturbation step.
        list_flexo_tensors (list): Flexoelectric tensors for each perturbation.
        list_piezo_tensors (list): Piezoelectric tensors for each perturbation.

    Methods:
        calculate_energy():
            Calculates the energy associated with the perturbation.
        plot_perturbation_data(save_plot=False, filename='perturbation_plot.png'):
            Plots the perturbation data for visualization and analysis.
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
        host_spec = 'mpirun -hosts=localhost -np 30'
    ):
        """
        Initializes the Perturbations instance with additional parameters.

        Args:
            abinit_file (str): Path to the Abinit file.
            min_amp (float): Minimum amplitude of perturbations.
            max_amp (float): Maximum amplitude of perturbations.
            perturbation (np.ndarray): Numpy array representing the perturbations.
            batch_script_header_file (str): Path to the batch script header file.
        """

        if not isinstance(perturbation, np.ndarray):
            raise ValueError("perturbation should be a numpy array.")

        # Initialize the "inputfile"
        super().__init__(
            name=name,
            num_datapoints=num_datapoints,
            abi_file=abi_file,
            min_amp=min_amp,
            max_amp=max_amp,
        )

        # Create an AbinitFile instance
        self.abinit_file = AbinitFile(
            abi_file=abi_file, batch_script_header_file=batch_script_header_file
        )

        self.pert = np.array(perturbation, dtype=np.float64)
        self.host_spec = str(host_spec)

        self.list_abi_files = []
        self.perturbed_objects = [self.abinit_file]
        self.list_energies = []
        self.list_amps = []
        self.list_flexo_tensors = []
        self.list_piezo_tensors_clamped = []
        self.list_piezo_tensors_relaxed = []

    def generate_perturbations(self):
        """
        Generates perturbed unit cells based on the given number of datapoints.

        Args:
            num_datapoints (int): Number of perturbed unit cells to generate.

        Returns:
            list: A list containing the perturbed unit cells.
        """

        # Calculate the step size
        step_size = (self.max_amp - self.min_amp) / (self.num_datapoints - 1)

        for i in range(self.num_datapoints):
            # Calculate the current amplitude factor
            current_amp = self.min_amp + i * step_size
            self.list_amps.append(current_amp)

            # Compute the perturbations
            perturbed_values = current_amp * self.pert
            perturbation_result = self.abinit_file.perturbations(perturbed_values)

            # Add the new object to a list
            self.perturbed_objects.append(perturbation_result)

    def calculate_energy_of_perturbations(self):
        """
        Runs an energy calculation for each of the Abinit perturbation objects.
        """
        for i, perturbation_object in enumerate(self.perturbed_objects):

            # Change file name for sorting when running energy_calculation batch
            perturbation_object.file_name = FlpzCore._get_unique_filename(
                f"{perturbation_object.file_name}_{i}"
            )
            perturbation_object.file_name = os.path.basename(perturbation_object.file_name)

            # Run energy calculation and save file name 
            perturbation_object.run_energy_calculation(host_spec=self.host_spec)
            self.list_abi_files.append(f"{perturbation_object.file_name}.abi")

            # Append most recent job to my array
            self.abinit_file.running_jobs.append(perturbation_object.running_jobs[-1])

        # Extract energy information
        self.abinit_file.wait_for_jobs_to_finish(check_time=90)
        for pertrubation_object in self.perturbed_objects:
            pertrubation_object.grab_energy()
            self.list_energies.append(pertrubation_object.energy)

    def calculate_piezo_of_perturbations(self):
        """
        Runs an energy and piezoelectric calculation for each of the Abinit perturbation objects.
        """
        for i, perturbation_object in enumerate(self.perturbed_objects):

            # Change the file name for sorting when running flexo_calculation batch
            perturbation_object.file_name = FlpzCore._get_unique_filename(
                f"{perturbation_object.file_name}_{i}"
            )
            perturbation_object.file_name = os.path.basename(perturbation_object.file_name)

            # Run piezo calculation and save file name 
            perturbation_object.run_piezo_calculation(host_spec=self.host_spec)
            self.list_abi_files.append(f"{perturbation_object.file_name}.abi")

            # Append most recent job to my array
            self.abinit_file.running_jobs.append(perturbation_object.running_jobs[-1])

            # Extract energy and piezoelectric properties 
            self.abinit_file.wait_for_jobs_to_finish(check_time=300)
            for perturbation_object in self.perturbed_objects:
                perturbation_object.grab_cartesian_coordinates()
                perturbation_object.grab_piezo_tensor()
                self.list_energies.append(perturbation_object.energy)
                self.list_piezo_tensors_clamped.append(
                    perturbation_object.piezo_tensor_clamped
                )
                self.list_piezo_tensors_relaxed.append(
                    perturbation_object.piezo_tensor_relaxed
                )

    def calculate_flexo_of_perturbations(self):
        """
        Runs an energy, piezoelectric, and flexoelectric calculation for each of the Abinit perturbation objects.
        """
        for i, perturbation_object in enumerate(self.perturbed_objects):

            # Change the file name for sorting when running flexo_calculation batch
            perturbation_object.file_name = FlpzCore._get_unique_filename(
                f"{perturbation_object.file_name}_{i}"
            )
            perturbation_object.file_name = os.path.basename(perturbation_object.file_name)

            # Run flexoelectricity calculation
            perturbation_object.run_flexo_calculation(host_spec=self.host_spec)
            self.list_abi_files.append(f"{perturbation_object.file_name}.abi")

            # Append most recent job to my array
            self.abinit_file.running_jobs.append(perturbation_object.running_jobs[-1])

        self.abinit_file.wait_for_jobs_to_finish(check_time=600)
        for perturbation_object in self.perturbed_objects:
            perturbation_object.grab_energy()
            perturbation_object.grab_flexo_tensor()
            perturbation_object.grab_piezo_tensor()
            self.list_energies.append(perturbation_object.energy)
            self.list_piezo_tensors_clamped.append(
                perturbation_object.piezo_tensor_clamped
            )
            self.list_piezo_tensors_relaxed.append(
                perturbation_object.piezo_tensor_relaxed
            )
            self.list_flexo_tensors.append(perturbation_object.flexo_tensor)

    # TODO: this function is not finised
    def data_analysis(
        self,
        piezo=False,
        flexo=False,
        save_plot=False,
        filename="energy_vs_amplitude.png",
        component_string="all",
        plot_piezo_relaxed_tensor=False,
    ):
        """
        Analyzes data by plotting energy or tensor components against amplitude,
        optionally saving plots.

        Args:
            piezo (bool): If True, analyzes and plots piezoelectric data.
            flexo (bool): If True, analyzes and plots flexoelectric data.
            save_plot (bool): If True, saves the plot to the current working directory.
            filename (str): The name of the file to save the plot as if save_plot is True.
            component_string (str): Specifies which tensor components to plot ('all' or specific indices).
        """

        if flexo:
            if len(self.list_amps) != len(self.list_flexo_tensors):
                raise ValueError(
                    "Mismatch between x_vec and list of flexoelectric tensors."
                )

            # Determine the number of components in the flattened tensor
            num_components = self.list_flexo_tensors[0].flatten().size

            if component_string == "all":
                selected_indices = list(range(num_components))
            else:
                try:
                    selected_indices = [int(i) - 1 for i in component_string.split()]
                    if any(i < 0 or i >= num_components for i in selected_indices):
                        raise ValueError
                except ValueError:
                    raise ValueError(
                        f"Invalid input in component_string. Please enter numbers between 1 and {num_components}."
                    )

            # Prepare data for plotting
            plot_data = np.zeros((len(self.list_flexo_tensors), len(selected_indices)))
            for idx, tensor in enumerate(self.list_flexo_tensors):
                flat_tensor = (
                    tensor.flatten()
                )  # Flatten the tensor from left to right, top to bottom
                plot_data[idx, :] = flat_tensor[selected_indices]

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))

            for i in range(len(selected_indices)):
                ax.plot(
                    self.list_amps,
                    plot_data[:, i],
                    linestyle=":",
                    marker="o",
                    markersize=8,
                    linewidth=1.5,
                    label=f"μ_{selected_indices[i] + 1}",
                )

            # Customize the plot
            ax.set_xlabel("x (bohrs)", fontsize=14)
            ax.set_ylabel(r"$\mu_{i,j} \left(\frac{nC}{m}\right)$", fontsize=14)
            ax.set_title("Flexoelectric Tensor Components vs. Amplitude of Displacement", fontsize=16)

            # Set axis limits as requested
            ax.set_xlim(0, self.max_amp)
            ax.set_ylim(0, np.max(plot_data))

            # Add grid, legend, and adjust layout
            ax.grid(True)
            ax.legend(loc="best", fontsize=12)
            ax.tick_params(axis="both", which="major", labelsize=14)


            # Adjust layout
            plt.tight_layout(pad=0.5)

            # Save the plot if required
            if save_plot:
                plt.savefig(f"{filename}_relaxed", bbox_inches="tight")
                print(f"Plot saved as {filename} in {os.getcwd()}")

            # Show the plot
            plt.show()

        elif piezo:
            if len(self.list_amps) != len(self.list_piezo_tensors_clamped):
                raise ValueError(
                    "Mismatch between x_vec and list of flexoelectric tensors."
                )

            # Determine the number of components in the flattened tensor
            num_components = self.list_piezo_tensors_clamped[0].flatten().size

            if component_string == "all":
                selected_indices = list(range(num_components))
            else:
                try:
                    selected_indices = [int(i) - 1 for i in component_string.split()]
                    if any(i < 0 or i >= num_components for i in selected_indices):
                        raise ValueError
                except ValueError:
                    raise ValueError(
                        f"Invalid input in component_string. Please enter numbers between 1 and {num_components}."
                    )

            # Prepare data for plotting
            plot_data_clamped = np.zeros(
                (len(self.list_piezo_tensors_clamped), len(selected_indices))
            )
            for idx, tensor in enumerate(self.list_piezo_tensors_clamped):
                flat_tensor = (
                    tensor.flatten()
                )  # Flatten the tensor from left to right, top to bottom
                plot_data_clamped[idx, :] = flat_tensor[selected_indices]

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))

            for i in range(len(selected_indices)):
                ax.plot(
                    self.list_amps,
                    plot_data_clamped[:, i],
                    linestyle=":",
                    marker="o",
                    markersize=8,
                    linewidth=1.5,
                    label=f"μ_{selected_indices[i] + 1}",
                )

                # Customize the plot
            ax.set_xlabel("x (bohrs)", fontsize=14)
            ax.set_ylabel(r"$\mu_{i,j} \left(\frac{nC}{m}\right)$", fontsize=14)
            ax.set_title("Piezoelectric Tensor Components (Clamped) vs. Amplitude of Displacement", fontsize=16)

            # Set axis limits as requested
            ax.set_xlim(0, self.max_amp)
            ax.set_ylim(0, np.max(plot_data_clamped))

            # Add grid, legend, and adjust layout
            ax.grid(True)
            ax.legend(loc="best", fontsize=12)
            ax.tick_params(axis="both", which="major", labelsize=14)

            # Adjust layout
            plt.tight_layout(pad=0.5)

            # Save the plot if required

            filename_based = os.path.basename(filename)
            filename = f"{filename_based}_clamped.png"

            if save_plot:
                plt.savefig(f"{filename}_relaxed", bbox_inches="tight")
                print(f"Plot saved as {filename} in {os.getcwd()}")

            # Show the plot
            plt.show()

            if plot_piezo_relaxed_tensor:

                # Prepare data for plotting
                plot_data_relaxed = np.zeros(
                    (len(self.list_piezo_tensors_relaxed), len(selected_indices))
                )
                for idx, tensor in enumerate(self.list_piezo_tensors_relaxed):
                    flat_tensor = (
                        tensor.flatten()
                    )  # Flatten the tensor from left to right, top to bottom
                    plot_data_relaxed[idx, :] = flat_tensor[selected_indices]

                # Create the plot
                fig, ax = plt.subplots(figsize=(8, 6))

                for i in range(len(selected_indices)):
                    ax.plot(
                        self.list_amps,
                        plot_data_relaxed[:, i],
                        linestyle=":",
                        marker="o",
                        markersize=8,
                        linewidth=1.5,
                        label=f"μ_{selected_indices[i] + 1}",
                    )

                    # Customize the plot
                ax.set_xlabel("x (bohrs)", fontsize=14)
                ax.set_ylabel(r"$\mu_{i,j} \left(\frac{nC}{m}\right)$", fontsize=14)
                ax.set_title("Piezoelectric Tensor Components (Relaxed) vs. Amplitude of Displacement", fontsize=16)

                # Set axis limits as requested
                ax.set_xlim(0, self.max_amp)
                ax.set_ylim(0, np.max(plot_data_relaxed))

                # Add grid, legend, and adjust layout
                ax.grid(True)
                ax.legend(loc="best", fontsize=12)
                ax.tick_params(axis="both", which="major", labelsize=14)

                # Adjust layout
                plt.tight_layout(pad=0.5)

                # Save the plot if required

                filename_based = os.path.basename(filename)
                filename = f"{filename_based}_relaxed.png"

                if save_plot:
                    plt.savefig(f"{filename}_relaxed", bbox_inches="tight")
                    print(f"Plot saved as {filename} in {os.getcwd()}")

                # Show the plot
                plt.show()


        else:

            if len(self.list_energies) != len(self.list_amps):
                raise ValueError(
                    "The length of list_energies and list_amps must be the same."
                )

            # Create the figure and axis
            fig, ax = plt.subplots()

            # Plot the points with lines
            ax.plot(
                self.list_amps, self.list_energies, marker="o", linestyle="-", color="b"
            )

            # Set title and labels
            ax.set_title("Energy vs Amplitude of Perturbations")
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("Energy")

            # Set axis limits using self.min and self.max for the x-axis
            x_margin = 0.1 * (
                self.max_amp - self.min_amp
            )  # Add margin for better visualization
            y_margin = 0.1 * (max(self.list_energies) - min(self.list_energies))
            ax.set_xlim(self.min_amp - x_margin, self.max_amp + x_margin)
            ax.set_ylim(
                min(self.list_energies) - y_margin, max(self.list_energies) + y_margin
            )

            # Enable grid
            ax.grid(True)

            # Adjust layout
            plt.tight_layout(pad=0.5)

            # Save the plot if required
            if save_plot:
                plt.savefig(filename, bbox_inches="tight")
                print(f"Plot saved as {filename} in {os.getcwd()}")

            # Show the plot
            plt.show()
