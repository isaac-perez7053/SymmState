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

# Set numpy error handling (temporary workaround for freq_thz issues)
np.seterr(all='ignore') 


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
        unstable_threshold=-20,
    ):
        """
        Initializes a SmodesProcessor with specified input file, SMODES parameters, and Abinit configurations.
        """
        # Initialize an AbinitFile using a symmetry informed basis.
        self.abinit_file = AbinitFile(
            abi_file=abi_file,
            smodes_input=smodes_input,
            target_irrep=target_irrep,
            batch_script_header_file=b_script_header_file,
            symmetry_informed_basis=True,
        )
        # Use the new _symmetry_adapted_basis method (updated name) to obtain SMODES output.
        _, smodes_output = AbinitFile._symmetry_adapted_basis(
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

        # Unstable threshold
        self.unstable_threshold = unstable_threshold

    def _loop_modes(self):
        """
        Creates the displaced cells and runs them through Abinit for the _perform_calculations method.
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
                self.abinit_file.coordinates_xcart + (1.88973 * self.disp_mag * self.dist_mat[i])
            )
            self.abinit_file.change_coordinates(
                new_coordinates=perturbation, coords_are_cartesian=True
            )
            abi_name = f"dist_{j}"
            batch_name = f"dist_{j}_sbatch"
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
            self.abinit_file.change_coordinates(
                np.array(original_coords).copy(), coords_are_cartesian=False
            )
            self.jobs_ran_abo.append(f"dist_{j}.abo")

        # Wait for jobs to finish
        self.abinit_file.wait_for_jobs_to_finish(60)

    def _perform_calculations(self, stabilize=False):
        """
        Calculates the eigen-frequencies associated with a particular representation.
        """
        # Ensure force_mat_raw is float32 (could be adjusted as needed)
        force_mat_raw = np.zeros((self.num_sam + 1, self.abinit_file.natom, 3), dtype=np.float32)

        for sam, abo in enumerate(self.jobs_ran_abo):
            with open(abo) as f:
                abo_lines = f.readlines()

            line_start = 0
            atom_ind = 0

            for line_num, line in enumerate(abo_lines):
                words = line.split()
                if len(words) >= 3 and words[0] == "cartesian" and words[1] == "forces" and words[2] == "(eV/Angstrom)":
                    line_start = line_num + 1
                    break

            for line_num in range(line_start, line_start + self.abinit_file.natom):
                words = abo_lines[line_num].split()
                force_mat_raw[sam, atom_ind, 0] = float(words[1])
                force_mat_raw[sam, atom_ind, 1] = float(words[2])
                force_mat_raw[sam, atom_ind, 2] = float(words[3])
                atom_ind += 1

        print(f"Printing force_mat_raw:\n{force_mat_raw}\n")

        # Subtract off the forces from the original cell
        force_list = np.zeros((self.num_sam, self.abinit_file.natom, 3), dtype=np.float32)
        for sam in range(self.num_sam):
            for i in range(self.abinit_file.natom):
                for j in range(3):
                    force_list[sam, i, j] = force_mat_raw[sam + 1, i, j] - force_mat_raw[0, i, j]

        print(f"Printing force list:\n{force_list}\n")

        # Compute force matrix via tensordot
        force_matrix = np.tensordot(force_list, self.dist_mat.astype(np.float64), axes=([1, 2], [1, 2]))
        self.force_matrix = np.array(force_matrix, dtype=np.float32)

        # Construct mass matrix
        mass_vector = np.zeros(self.num_sam, dtype=np.float32)
        for m in range(self.num_sam):
            this_mass = 0
            for n in range(self.abinit_file.ntypat):
                if self.sam_atom_label[m] == self.type_list[n]:
                    this_mass = self.mass_list[n]
                    mass_vector[m] = this_mass
            if this_mass == 0:
                raise ValueError("Problem with building mass matrix. Quitting...")

        print(f"Mass Vector:\n{mass_vector}\n")
        sqrt_mass_vector = np.sqrt(mass_vector)
        mass_matrix = np.outer(sqrt_mass_vector, sqrt_mass_vector)
        print(f"Mass Matrix:\n{mass_matrix}\n")
        self.mass_matrix = np.array(mass_matrix, dtype=np.float32)

        # Compute force constants matrix
        fc_mat = (-force_matrix / self.disp_mag).astype(np.float32)
        fc_mat = (fc_mat + fc_mat.T) / 2.0
        self.springs_constants_matrix = fc_mat
        print(f"Force Constants Matrix:\n{self.springs_constants_matrix}\n")

        cond_number = np.linalg.cond(fc_mat)
        print(f"Condition number of the force constants matrix: {cond_number}\n")
        if cond_number > 1e5:
            warnings.warn("High numerical instability in force constants matrix.")

        fc_evals, _ = np.linalg.eig(fc_mat)

        dyn_mat = np.divide(fc_mat, mass_matrix)
        self.dyn_mat = dyn_mat
        print(f"Dynamical Matrix:\n{dyn_mat}\n")

        cond_number = np.linalg.cond(dyn_mat)
        print(f"Condition number of the dynamical matrix: {cond_number}\n")
        if cond_number > 1e5:
            warnings.warn("High numerical instability in dynamical matrix.")

        dynevals, dynevecs_sam = np.linalg.eig(dyn_mat)
        print(f"DEBUG: dynevecs_sam:\n{dynevecs_sam}\n")
        print(f"DEBUG: dynevals:\n{dynevals}\n")
        print("Eigenvalues:", dynevals)
        print("Absolute eigenvalues:", np.abs(dynevals))

        # Frequency conversion factors
        eV_to_J = 1.602177E-19
        ang_to_m = 1.0E-10
        AMU_to_kg = 1.66053E-27
        c = 2.9979458E10  # speed of light

        freq_thz = np.sign(dynevals) * np.sqrt(np.abs(dynevals) * eV_to_J / (ang_to_m**2 * AMU_to_kg)) * 1.0E-12
        fc_eval = np.sign(fc_evals) * np.sqrt(np.abs(fc_evals))
        print(f"DEBUG: freq_thz:\n{freq_thz}\n")
        idx_dyn = np.flip(np.argsort(freq_thz)[::-1])
        freq_thz = freq_thz[idx_dyn] / (2 * np.pi)
        dynevecs_sam = dynevecs_sam[:, idx_dyn]
        freq_cm = freq_thz * 1.0E12 / c
        print(f"Frequency in wavenumbers: {freq_cm}")

        self.dyn_freqs = [[freq_thz[i], freq_cm[i]] for i in range(self.num_sam)]
        self.fc_evals = fc_eval[idx_dyn]

        dynevecs = np.zeros((self.num_sam, self.abinit_file.natom, 3), dtype=np.float32)
        for evec in range(self.num_sam):
            real_dynevec = np.zeros((self.abinit_file.natom, 3), dtype=np.float32)
            for s in range(self.num_sam):
                real_dynevec += dynevecs_sam[s, evec] * self.dist_mat[s, :, :]
            dynevecs[evec, :, :] = real_dynevec

        print(f"DEBUG: Dynevecs:\n{dynevecs}\n")

        mass_col = np.zeros((self.abinit_file.natom, 3), dtype=np.float32)
        atomind = 0
        for atype in range(self.abinit_file.ntypat):
            for j in range(self.type_count[atype]):
                mass_col[atomind, :] = np.sqrt(self.mass_list[atype])
                atomind += 1

        phon_disp_eigs = np.zeros((self.num_sam, self.abinit_file.natom, 3), dtype=np.float32)
        redmass_vec = np.zeros((self.abinit_file.natom, 1), dtype=np.float32)
        for mode in range(self.num_sam):
            phon_disp_eigs[mode, :, :] = np.divide(dynevecs[mode, :, :], mass_col)
            mag_squared = np.sum(phon_disp_eigs[mode, :, :]**2)
            redmass_vec[mode] = 1.0 / mag_squared
            phon_disp_eigs[mode, :, :] /= np.sqrt(mag_squared)

        self.phonon_vecs = phon_disp_eigs.astype(np.float64)
        self.red_mass = redmass_vec.astype(np.float32)
        print(f"DEBUG: Reduced mass vector:\n{self.red_mass}\n")
        print("Computation completed. Results stored in object attributes.")

    def _perform_calculations_dfpt(self):
        pass

    def _imaginary_frequencies(self):
        negative_indices = []
        print(f"DEBUG: Phonon vectors:\n{self.phonon_vecs}\n")
        for index, fc_eval in enumerate(self.fc_evals):
            if fc_eval < self.unstable_threshold:
                negative_indices.append(index)
        print(f"DEBUG: Unstable indices:\n{negative_indices}\n")
        return negative_indices if negative_indices else False

    def stabilize_matrix(self, matrix, threshold=50000, epsilon=1e-12, alpha=0.001):
        initial_cond_number = np.linalg.cond(matrix)
        print(f"Initial Condition Number: {initial_cond_number}\n")
        if initial_cond_number > threshold:
            print("Applying stabilization.")
            initial_diagonal = np.diag(matrix).copy()
            for i in range(matrix.shape[0]):
                row_sum = np.sum(np.abs(matrix[i, :])) - matrix[i, i]
                if matrix[i, i] < row_sum:
                    matrix[i, i] = (1 - epsilon) * initial_diagonal[i] + epsilon * row_sum
            sym_matrix = (matrix + matrix.T) / 2
            matrix = (1 - alpha) * matrix + alpha * sym_matrix
        stabilized_cond_number = np.linalg.cond(matrix)
        print(f"Stabilized Condition Number: {stabilized_cond_number}\n")
        return matrix

    def run_smodes(self, smodes_input):
        if not Path(self.smodes_path).is_file():
            raise FileNotFoundError(f"SMODES executable not found at: {self.smodes_path}")
        command = f"{self.smodes_path} < {smodes_input} > output.log"
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"SMODES execution failed: {process.stderr}")
        return process.stdout

    def unstable_phonons(self):
        unstable_normalized = []
        unstable = self._imaginary_frequencies()
        if unstable is False:
            print("No unstable phonons present")
            return False
        else:
            for i in unstable:
                flattened = self.phonon_vecs[i].flatten()
                norm_val = np.linalg.norm(flattened)
                normalized = flattened / norm_val
                normalized_matrix = normalized.reshape(self.phonon_vecs[i].shape)
                unstable_normalized.append(normalized_matrix)
            print(f"DEBUG: Normalized unstable phonons:\n{unstable_normalized}")
            return unstable_normalized

    def symmadapt(self):
        self._loop_modes()
        self._perform_calculations()
        return self.unstable_phonons()


def main():
    input_file = str(sys.argv[1])
    smodesInput = str(sys.argv[2])
    target_irrep = str(sys.argv[3])
    calculator = SmodesProcessor(input_file, smodesInput, target_irrep)
    # Uncomment the following to print outputs:
    # print("Dyn Frequencies (THz, cm^-1):", calculator.dyn_freqs)
    # print("Force Constant Evals:", calculator.fc_evals)
    # print("Phonon Vecs Shape:", calculator.phonon_vecs.shape)
    # print("Reduced Masses:", calculator.red_mass)


if __name__ == "__main__":
    main()
