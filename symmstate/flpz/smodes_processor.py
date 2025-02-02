import numpy as np
import subprocess
import sys
from pathlib import Path
from scipy.sparse.linalg import cg
import warnings
from symmstate.abinit import AbinitFile
from symmstate.flpz import FlpzCore

np.set_printoptions(precision=10)

import tracemalloc

tracemalloc.start()


# Ultimate TODO: Port SmodesProcessor into the programs section and plan for it to be ran by other programs
class SmodesProcessor(FlpzCore):
    """
    A class that processes symmetry modes (SMODES) to calculate phonon properties
    and analyzes them using Abinit simulations.

    Attributes:
        target_irrep (str): The irreducible representation targeted in the calculations.
        smodes_path (str): The path to the SMODES executable.
        symm_prec (float): Precision for recognizing symmetry operations.
        disp_mag (float): Magnitude of displacements used in calculations.
        host_spec (str): Specifications used for parallel execution.
        irrep (str): Irreducible representation identifier.
        num_sam (int): Number of systematic atomic modes (SAMs).
        sam_atom_label (list): Labels of atoms in SAMs.
        mass_list (list): List of atomic masses.
        dist_mat (np.ndarray): Displacement matrix for SAMs.
        pos_mat_cart (np.ndarray): Cartesian coordinates of atomic positions.
        type_count (list): Count of each atom type.
        isir (bool): Indicates if infrared active modes are present.
        israman (bool): Indicates if Raman active modes are present.
        transmodes (bool): Indicates if translational modes exist.
        crossdot_ispos (bool): If cross product of rprim vectors is positive.
        mass_matrix (np.ndarray): Matrix containing atomic masses.
        springs_constants_matrix (np.ndarray): Spring constants matrix.
        jobs_ran_abo (list): List of job identifiers processed by Abinit.
        dyn_freqs (list): List of dynamic frequencies.
        fc_evals (np.ndarray): Eigenvalues of the force constant matrix.
        phonon_vecs (np.ndarray): Phonon displacement vectors.
        red_mass (np.ndarray): Reduced masses for each mode.

    Public Methods:
        convertToXcart(): Converts and returns Cartesian coordinates of the unit cell.
        convertToXred(): Converts and returns reduced coordinates of the unit cell.
        findSpaceGroup(): Determines and returns the space group of the unit cell.
        write_custom_abifile(output_file, header_file): Writes a custom Abinit .abi file.
        run_smodes(smodes_input): Executes SMODES and processes its output.
        unstable_phonons(): Outputs unstable phonons or indicates their absence.
        symmadapt(): Runs the symmadapt sub-program for symmetry-related modes adaptation.
    """

    def __init__(
        self,
        abi_file=None,
        smodes_input=None,
        target_irrep=None,
        smodes_path="../isobyu/smodes",
        host_spec="mpirun -hosts=localhost -np 20",
        disp_mag=0.001,
        symm_prec=1e-5,
        b_script_header_file=None,
    ):
        """
        Initializes a SmodesProcessor with specified input file, SMODES parameters, and Abinit configurations.

        Args:
            abi_file (str): Path to the Abinit input file.
            smodes_input (str): Path to the SMODES input file.
            target_irrep (str): The target irreducible representation.
            symm_prec (float): Symmetry precision for identifying symmetry operations. Defaults to 1e-5.
            disp_mag (float): Displacement magnitude for calculations. Defaults to 0.001.
            smodes_path (str): Executable path for SMODES. Defaults to '../isobyu/smodes'.
            host_spec (str): Specification string for host setup in distributed computing. Defaults to 'mpirun -hosts=localhost -np 30'.
            b_script_header_file (str, optional): Path to batch script header file. Defaults to None.
        """
        # Initialize an Abinit File using a symmetry informed basis
        self.abinit_file = AbinitFile(
            abi_file=abi_file,
            smodes_input=smodes_input,
            target_irrep=target_irrep,
            batch_script_header_file=b_script_header_file,
            symmetry_informed_basis=True,
        )
        _, smodes_output = AbinitFile._symmatry_adapted_basis(
            smodes_file=smodes_input,
            target_irrep=target_irrep,
            symm_prec=symm_prec,
            smodes_path=smodes_path,
        )

        self.transmodes = smodes_output[0]
        self.isir = smodes_output[1]
        self.israman = smodes_output[2]
        self.type_count = smodes_output[3]
        self.type_list = smodes_output[4]
        self.num_sam = smodes_output[5]
        self.mass_list = smodes_output[6]
        self.pos_mat_cart = smodes_output[7]
        self.dist_mat = smodes_output[8]
        self.sam_atom_label = smodes_output[9]

        self.smodes_path = smodes_path

        self.disp_mag = disp_mag
        self.host_spec = host_spec

        self.mass_matrix = None
        self.force_matrix = None
        self.springs_constants_matrix = None
        self.dyn_mat = None

        # Attributes for storing calculated data
        self.jobs_ran_abo = []
        self.dyn_freqs = None
        self.fc_evals = None
        self.phonon_vecs = None
        self.red_mass = None

    def _loop_modes(self):
        """
        Creates the displaced cells and runs them through Abinit
        for the _perform_calculations method.
        """
        content = """
useylm 1
kptopt 2
chkprim 0
"""
        original_coords = self.abinit_file.coordinates_xred.copy()
        abi_name = "dist_0"
        batch_name = "dist_0_sbatch"
        self.abinit_file.write_custom_abifile(
            abi_name, content, coords_are_cartesian=False
        )
        self.abinit_file.run_abinit(
            input_file=abi_name,
            batch_name=batch_name,
            batch_script_header_file=self.abinit_file.batch_header,
            host_spec=self.host_spec,
            log="dist_0.log",
            delete_batch_script=False,
        )
        self.jobs_ran_abo.append(f"dist_0.abo")

        # Displace each cell
        for i in range(self.num_sam):
            j = i + 1

            # Calculate displacement
            perturbation = np.array(
                self.abinit_file.coordinates_xcart
                + (1.88973 * self.disp_mag * self.dist_mat[i])
            )
            self.abinit_file.change_coordinates(
                new_coordinates=perturbation, coords_are_cartesian=True
            )
            abi_name = f"dist_{j}"
            batch_name = f"dist_{j}_sbatch"

            # Write abifile and run abinit
            self.abinit_file.write_custom_abifile(
                abi_name, content, coords_are_cartesian=False
            )
            self.abinit_file.run_abinit(
                input_file=abi_name,
                batch_name=batch_name,
                batch_script_header_file=self.abinit_file.batch_header,
                host_spec=self.host_spec,
                log=f"dist_{j}.log",
                delete_batch_script=False,
            )

            # Change coordinates back to their original value
            self.abinit_file.change_coordinates(
                np.array(original_coords).copy(), coords_are_cartesian=False
            )
            self.jobs_ran_abo.append(f"dist_{j}.abo")

        # Wait for jobs to finish
        self.abinit_file.wait_for_jobs_to_finish(60)

    def _perform_calculations(self, stabilize=False):
        """
        Calculates the eigen-frequencies associated with a particular representation.

        Raises:
            ValueError: If preconditions like positive cross-product are not met.
        """
        # Ensure force_mat_raw is float64
        force_mat_raw = np.zeros(
            (self.num_sam + 1, self.abinit_file.natom, 3), dtype=np.float32
        )

        for sam, abo in enumerate(self.jobs_ran_abo):
            with open(abo) as f:
                abo_lines = f.readlines()

            line_start = 0
            atom_ind = 0

            for line_num, line in enumerate(abo_lines):
                words = line.split()
                if (
                    len(words) >= 1
                    and words[0] == "cartesian"
                    and words[1] == "forces"
                    and words[2] == "(eV/Angstrom)"
                ):
                    line_start = line_num + 1
                    break

            for line_num in range(line_start, line_start + self.abinit_file.natom):
                words = abo_lines[line_num].split()
                force_mat_raw[sam, atom_ind, 0] = float(words[1])
                force_mat_raw[sam, atom_ind, 1] = float(words[2])
                force_mat_raw[sam, atom_ind, 2] = float(words[3])
                atom_ind += 1

        print(f"Printing force_mat_raw:\n \n {force_mat_raw} \n")

        # Create force_list with dtype float64
        force_list = np.zeros(
            (self.num_sam, self.abinit_file.natom, 3), dtype=np.float32
        )

        # Subtract off the forces from the original cell
        for sam in range(self.num_sam):
            for i in range(self.abinit_file.natom):
                for j in range(3):
                    force_list[sam, i, j] = (
                        force_mat_raw[sam + 1, i, j] - force_mat_raw[0, i, j]
                    )

        print(f"Printing force list: \n \n {force_list} \n")

        # Initialize the force matrix with dtype float64
        force_matrix = np.zeros((self.num_sam, self.num_sam), dtype=np.float32)

        # Vectorized computation of the force matrix
        force_matrix = np.tensordot(
            force_list, self.dist_mat.astype(np.float64), axes=([1, 2], [1, 2])
        )

        # Store the force matrix
        self.force_matrix = np.array(force_matrix, dtype=np.float32)

        # Construct the mass matrix #
        #############################

        # Initialize the mass vector
        mass_vector = np.zeros(self.num_sam, dtype=np.float32)

        # Build the mass vector
        for m in range(self.num_sam):
            this_mass = 0
            for n in range(self.abinit_file.ntypat):
                if self.sam_atom_label[m] == self.type_list[n]:
                    this_mass = self.mass_list[n]
                    mass_vector[m] = this_mass
            if this_mass == 0:
                raise ValueError("Problem with building mass matrix. Quitting...")

        # Print the mass vector for debugging
        print(f"Mass Vector:\n{mass_vector}\n")

        # Compute the square root of the mass vector
        sqrt_mass_vector = np.sqrt(mass_vector)

        # Fill the mass matrix using an outer product
        mass_matrix = np.outer(sqrt_mass_vector, sqrt_mass_vector)

        # Print the initial mass matrix
        print(f"Mass Matrix:\n{mass_matrix}\n")

        # Store the mass matrix
        self.mass_matrix = np.array(mass_matrix, dtype=np.float32)

        # Construct the fc_mat matrix #
        ###############################
        fc_mat = (-force_matrix / self.disp_mag).astype(np.float32)
        fc_mat = (fc_mat + np.transpose(fc_mat)) / 2.0
        self.springs_constants_matrix = fc_mat

        print(f"Force Constants Matrix: \n {self.springs_constants_matrix} \n")

        cond_number = np.linalg.cond(fc_mat)
        print(f"Condition number of the force constants matrix: {cond_number}\n")
        if cond_number > 1e5:
            warnings.warn(
                "The numerical instability of the force constants matrix is high and may affect the reliability of its eigenvalues"
            )

        fc_evals, _ = np.linalg.eig(fc_mat)

        # Construct the dyn_mat matrix
        ###############################

        dyn_mat = np.divide(fc_mat, mass_matrix)

        # Print the initial dyn_mat
        self.dyn_mat = dyn_mat
        print(f"Dynamical Matrix:\n {dyn_mat} \n")

        cond_number = np.linalg.cond(dyn_mat)
        print(f"Condition number of the dynamical matrix: {cond_number}\n")
        if cond_number > 1e5:
            warnings.warn(
                "The numerical instability of the dynamical matrix is high and may affect the reliability of its eigenvalues"
            )

        dynevals, dynevecs_sam = np.linalg.eig(dyn_mat)

        print(f"DEBUG: Printing dynevecs_sam: \n {dynevecs_sam} \n")
        print(f"DEBUG: Printing dynevals: \n {dynevals} \n")

        eV_to_J = 1.602177e-19
        ang_to_m = 1.0e-10
        AMU_to_kg = 1.66053e-27
        c = 2.9979458e10

        freq_thz = np.multiply(
            np.sign(dynevals),
            np.sqrt(np.absolute(dynevals) * eV_to_J / (ang_to_m**2 * AMU_to_kg))
            * 1.0e-12,
        )
        fc_eval = np.multiply(np.sign(fc_evals), np.sqrt(np.absolute(fc_evals)))

        print(f"DEBUG: Printing freq_thz: \n {freq_thz} \n ")

        idx_dyn = np.flip(freq_thz.argsort()[::-1])

        print(f"DEBUG: Printing idx_dyn, \n {idx_dyn} \n")

        freq_thz = freq_thz[idx_dyn] / (2 * np.pi)
        dynevecs_sam = dynevecs_sam[:, idx_dyn]

        freq_cm = freq_thz * 1.0e12 / c

        self.dyn_freqs = [[freq_thz[i], freq_cm[i]] for i in range(self.num_sam)]
        self.fc_evals = fc_eval[idx_dyn]

        dynevecs = np.zeros((self.num_sam, self.abinit_file.natom, 3), dtype=np.float32)

        for evec in range(self.num_sam):
            real_dynevec = np.zeros((self.abinit_file.natom, 3), dtype=np.float32)
            for s in range(self.num_sam):
                real_dynevec += dynevecs_sam[s, evec] * self.dist_mat[s, :, :]
            dynevecs[evec, :, :] = real_dynevec

        print(f"DEGBUG: Printing Dynevecs: \n {dynevecs} \n")

        mass_col = np.zeros((self.abinit_file.natom, 3), dtype=np.float32)
        atomind = 0

        for atype in range(self.abinit_file.ntypat):
            for j in range(self.type_count[atype]):
                mass_col[atomind, :] = np.sqrt(self.mass_list[atype])
                atomind += 1

        phon_disp_eigs = np.zeros(
            (self.num_sam, self.abinit_file.natom, 3), dtype=np.float32
        )
        redmass_vec = np.zeros((self.abinit_file.natom, 1), dtype=np.float32)

        for mode in range(self.num_sam):
            phon_disp_eigs[mode, :, :] = np.divide(dynevecs[mode, :, :], mass_col)
            mag_squared = np.sum(
                np.sum(
                    np.multiply(phon_disp_eigs[mode, :, :], phon_disp_eigs[mode, :, :])
                )
            )
            redmass_vec[mode] = 1.0 / mag_squared
            phon_disp_eigs[mode, :, :] /= np.sqrt(mag_squared)

        self.phonon_vecs = phon_disp_eigs
        # Assign phonon vectors and reduced mass to object attributes
        self.red_mass = redmass_vec.astype(np.float32)

        print(f"DEBUG: Printing reduced mass vector: \n {self.red_mass} \n")

        # Store the phonon eigenvectors
        self.phonon_vecs = phon_disp_eigs.astype(np.float64)

        print(
            "Computation completed. The resulting matrices and vectors are stored in the object's attributes."
        )

    def _perform_calculations_dfpt(self):
        pass

    def _imaginary_frequencies(self):
        """
        Detects whether imaginary frequencies exist in dynamic modes.

        Returns:
            list/bool: List of indices with imaginary frequencies, or False if none.
        """
        # List to store indicies with negative freq_thz values
        negative_indicies = []
        print(f"DEBUG: Printing phonons: \n {self.phonon_vecs} \n")
        # Iterate over dyn_freqs to check the first element of each sublist
        for index, (_, freq_thz) in enumerate(self.dyn_freqs):
            if freq_thz < -20:
                negative_indicies.append(index)

        # Return the list of indicies or False if none are negative
        print(
            f"DEBUG: Printing unstable un normalized phonons: \n {negative_indicies} \n"
        )
        return negative_indicies if negative_indicies else False

    def stabilize_matrix(self, matrix, threshold=50000, epsilon=1e-12, alpha=0.001):
        """
        Stabilize a matrix by regularizing the diagonal, applying weighted symmetrization,
        and adjusting eigenvalues for numerical stability.

        Parameters:
            matrix (numpy.ndarray): The matrix to stabilize.
            threshold (float): Condition number threshold for stabilization.
            epsilon (float): Minimal regularization term for diagonal adjustment.
            alpha (float): Weight for symmetrization to preserve original values.

        Returns:
            numpy.ndarray: The stabilized matrix.
        """
        # Compute the initial condition number
        initial_cond_number = np.linalg.cond(matrix)
        print(f"Initial Condition Number of Matrix: {initial_cond_number}\n")

        if initial_cond_number > threshold:
            print("Condition number is too high; applying stabilization.")

            # Preserve diagonal values while improving stability
            initial_diagonal = np.diag(matrix).copy()

            # Regularize the diagonal minimally
            for i in range(matrix.shape[0]):
                row_sum = np.sum(np.abs(matrix[i, :])) - matrix[i, i]
                if matrix[i, i] < row_sum:
                    matrix[i, i] = (1 - epsilon) * initial_diagonal[
                        i
                    ] + epsilon * row_sum

            # Weighted symmetrization to balance original values and numerical stability
            symmetrized_matrix = (matrix + matrix.T) / 2
            matrix = (1 - alpha) * matrix + alpha * symmetrized_matrix

        # Compute the stabilized condition number
        stabilized_cond_number = np.linalg.cond(matrix)
        print(f"Stabilized Condition Number of Matrix: {stabilized_cond_number}\n")

        return matrix

    def run_smodes(self, smodes_input):
        """
        Run the SMODES executable and process its output.

        Args:
            smodes_input (str): Path to SMODES input file.

        Returns:
            str: Output captured from SMODES execution.

        Raises:
            FileNotFoundError: If SMODES executable is not found.
            RuntimeError: If SMODES execution fails.
        """
        if not Path(self.smodes_path).is_file():
            raise FileNotFoundError(
                f"SMODES executable not found at: {self.smodes_path}"
            )

        # Redirect output to the designated output directory
        command = f"{self.smodes_path} < {smodes_input} > output.log"
        process = subprocess.run(command, shell=True, capture_output=True, text=True)

        if process.returncode != 0:
            raise RuntimeError(f"SMODES execution failed: {process.stderr}")

        return process.stdout

    # TODO: check the vectors are normalized correctly
    def unstable_phonons(self):
        """
        Outputs the unstable phonons and false if none are present.

        Returns:
            list/bool: List of normalized matrices representing unstable phonons,
                       or False if none are detected.
        """
        unstable_phonons_normalized = []
        if self._imaginary_frequencies() == False:
            print("No unstable phonons are present")
            return False
        else:
            for i in self._imaginary_frequencies():
                # Normalize the matrix as if it were a single vector.
                flattened = self.phonon_vecs[i].flatten()
                euclidean_norm = np.linalg.norm(flattened)
                normalized_flattened = flattened / euclidean_norm
                normalized_matrix = normalized_flattened.reshape(
                    self.phonon_vecs[i].shape
                )
                unstable_phonons_normalized.append(normalized_matrix)
            print(
                f"DEBUG: Printing normalized unstable phonons: \n {unstable_phonons_normalized}"
            )
            return unstable_phonons_normalized

    def symmadapt(self):
        """
        Runs the symmadapt sub-program to determine and adapt symmetry-related modes.

        Returns:
            list/bool: Result of unstable phonons calculation.
        """
        self._loop_modes()
        self._perform_calculations()
        return self.unstable_phonons()


def main():
    input_file = str(sys.argv[1])
    smodesInput = str(sys.argv[2])
    target_irrep = str(sys.argv[3])  # Example usage with command-line argument
    calculator = SmodesProcessor(input_file, smodesInput, target_irrep)

    # print("Dyn Frequencies (THz, cm^-1):", calculator.dyn_freqs)
    # print("Force Constant Evals:", calculator.fc_evals)
    # print("Phonon Vecs Shape:", calculator.phonon_vecs.shape)
    # print("Reduced Masses:", calculator.red_mass)


# Usage example
if __name__ == "__main__":
    main()
