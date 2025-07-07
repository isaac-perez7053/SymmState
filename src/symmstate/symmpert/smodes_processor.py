import numpy as np
import subprocess
from pathlib import Path
import warnings
from symmstate.abinit import AbinitFile
from symmstate.config.symm_state_settings import settings
from symmstate.utils.symmetry_adapted_basis import SymmAdaptedBasis
from symmstate.slurm import SlurmFile

np.set_printoptions(precision=10)
import tracemalloc

tracemalloc.start()

np.seterr(all="ignore")


class SmodesProcessor:
    """
    A class that processes symmetry modes (SMODES) to calculate phonon properties
    and analyzes them using Abinit simulations.

    Attributes:
        target_irrep (str): The irreducible representation targeted in the calculations.
        disp_mag (float): Magnitude of displacements used in calculations.
        symm_prec (float): Precision for recognizing symmetry operations.
        unstable_threshold (int/float): Threshold below which phonon modes are considered unstable.
        slurm_obj: A SlurmFile instance used for job submission.
        (Other attributes such as num_sam, mass_list, dist_mat, etc. are extracted from SMODES output.)
    """

    def __init__(
        self,
        abi_file: str = None,
        smodes_input: str = None,
        target_irrep: str = None,
        disp_mag: float = 0.001,
        unstable_threshold: float = -20,
        slurm_obj: SlurmFile = None,
    ):
        """
        Initializes a SmodesProcessor with specified input file, SMODES parameters, and Abinit configurations.
        Uses the global SMODES_PATH from settings and a provided SlurmFile instance.
        """
        # Initialize an AbinitFile using a symmetry-informed basis.
        self.abinit_file = AbinitFile(
            abi_file=abi_file,
            smodes_input=smodes_input,
            target_irrep=target_irrep,
        )
        # Use the _symmetry_adapted_basis method to obtain SMODES output.
        _, smodes_output = SymmAdaptedBasis.symmatry_adapted_basis(
            smodes_file=smodes_input, target_irrep=target_irrep
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

        self.disp_mag = disp_mag
        self.symm_prec = settings.SYMM_PREC

        # Use the slurm_obj for job management.
        self.slurm_obj = slurm_obj

        # Attributes for storing calculated data.
        self.mass_matrix = None
        self.force_matrix = None
        self.springs_constants_matrix = None
        self.dyn_mat = None
        self.jobs_ran_abo = []  # Maintained locally for _loop_modes.
        self.dyn_freqs = None
        self.fc_evals = None
        self.phonon_vecs = None
        self.red_mass = None
        self.freq_cm = None

        self.unstable_threshold = unstable_threshold

    def _loop_modes(self):
        """
        Creates the displaced cells and runs them through Abinit for calculations.
        Job submissions are handled by the provided slurm_obj.
        """
        content = "useylm 1\nkptopt 2\nchkprim 0\n chksymtnons 0\n "
        original_coords = self.abinit_file.grab_reduced_coordinates()
        abi_name = "dist_0"
        self.abinit_file.write_custom_abifile(
            abi_name, content, coords_are_cartesian=False
        )
        self.abinit_file.run_abinit(
            input_file=abi_name,
            slurm_obj=self.slurm_obj,
            batch_name="dist_0_sbatch",
            log_file="dist_0.log",
        )
        # Record the job using the slurm_obj.
        self.jobs_ran_abo.append(f"dist_0.abo")

        for i in range(self.num_sam):
            j = i + 1
            self._logger.info(
                f"Printing number of symmetry adapted modes:\n {np.array2string(self.num_sam, precision=6, suppress_small=True)} \n"
            )
            perturbation = np.array(
                self.abinit_file.grab_cartesian_coordinates()
                + (1.88973 * self.disp_mag * self.dist_mat[i])
            )
            self._logger.info(
                f"Printing perturbation: \n {np.array2string(perturbation, precision=6, suppress_small=True)} \n"
            )
            self.abinit_file.change_coordinates(
                new_coordinates=perturbation, coords_are_cartesian=True
            )
            abi_name = f"dist_{j}"
            self.abinit_file.write_custom_abifile(
                abi_name, content, coords_are_cartesian=False
            )
            self.abinit_file.run_abinit(
                input_file=abi_name,
                slurm_obj=self.slurm_obj,
                batch_name=f"dist_{j}_sbatch",
                log_file=f"dist_{j}.log",
            )
            self.abinit_file.change_coordinates(
                np.array(original_coords).copy(), coords_are_cartesian=False
            )
            self.jobs_ran_abo.append(f"dist_{j}.abo")

        # Wait for all jobs to finish.
        self.slurm_obj.wait_for_jobs_to_finish(check_time=60)

    def _perform_calculations(self):
        """
        Calculates the eigen-frequencies associated with a particular representation.
        All diagnostic messages are logged.
        """
        self._logger.info("Starting _perform_calculations \n")
        force_mat_raw = np.zeros(
            (self.num_sam + 1, self.abinit_file.vars["natom"], 3), dtype=np.float64
        )

        for sam, abo in enumerate(self.jobs_ran_abo):
            with open(abo) as f:
                abo_lines = f.readlines()

            line_start = 0
            atom_ind = 0
            for line_num, line in enumerate(abo_lines):
                words = line.split()
                if (
                    len(words) >= 3
                    and words[0] == "cartesian"
                    and words[1] == "forces"
                    and words[2] == "(eV/Angstrom)"
                ):
                    line_start = line_num + 1
                    break

            for line_num in range(
                line_start, line_start + self.abinit_file.vars["natom"]
            ):
                words = abo_lines[line_num].split()
                force_mat_raw[sam, atom_ind, 0] = float(words[1])
                force_mat_raw[sam, atom_ind, 1] = float(words[2])
                force_mat_raw[sam, atom_ind, 2] = float(words[3])
                atom_ind += 1

        self._logger.info(
            f"Force matrix raw:\n{np.array2string(force_mat_raw, precision=6, suppress_small=True)} \n"
        )

        force_list = np.zeros(
            (self.num_sam, self.abinit_file.vars["natom"], 3), dtype=np.float64
        )
        for sam in range(self.num_sam):
            for i in range(self.abinit_file.vars["natom"]):
                for j in range(3):
                    force_list[sam, i, j] = (
                        force_mat_raw[sam + 1, i, j] - force_mat_raw[0, i, j]
                    )

        self._logger.info(
            f"Force list:\n{np.array2string(force_list, precision=6, suppress_small=True)} \n"
        )

        force_matrix = np.tensordot(
            force_list, self.dist_mat.astype(np.float64), axes=([1, 2], [1, 2])
        )
        self.force_matrix = np.array(force_matrix, dtype=np.float64)

        mass_vector = np.zeros(self.num_sam, dtype=np.float64)
        for m in range(self.num_sam):
            this_mass = 0
            for n in range(self.abinit_file.vars["ntypat"]):
                if self.sam_atom_label[m] == self.type_list[n]:
                    this_mass = self.mass_list[n]
                    mass_vector[m] = this_mass
            if this_mass == 0:
                raise ValueError("Problem with building mass matrix. Quitting...")
        self._logger.info(
            f"Mass Vector:\n{np.array2string(mass_vector, precision=6, suppress_small=True)} \n"
        )

        sqrt_mass_vector = np.sqrt(mass_vector)
        mass_matrix = np.outer(sqrt_mass_vector, sqrt_mass_vector)
        self._logger.info(
            f"Mass Matrix:\n{np.array2string(mass_matrix, precision=6, suppress_small=True)}\n"
        )

        fc_mat = (-force_matrix / self.disp_mag).astype(np.float64)
        fc_mat = (fc_mat + fc_mat.T) / 2.0
        self.springs_constants_matrix = fc_mat
        self._logger.info(
            f"Force Constants Matrix:\n{np.array2string(fc_mat, precision=6, suppress_small=True)} \n"
        )

        cond_number = np.linalg.cond(fc_mat)
        self._logger.info(
            f"Condition number of the force constants matrix: {cond_number}"
        )
        if cond_number > 1e5:
            warnings.warn("High numerical instability in force constants matrix.")

        fc_evals, _ = np.linalg.eig(fc_mat)

        dyn_mat = np.divide(fc_mat, mass_matrix)
        self.dyn_mat = dyn_mat
        self._logger.info(
            f"Dynamical Matrix:\n{np.array2string(dyn_mat, precision=6, suppress_small=True)} \n"
        )

        cond_number = np.linalg.cond(dyn_mat)
        self._logger.info(f"Condition number of the dynamical matrix: {cond_number}")
        if cond_number > 1e5:
            warnings.warn("High numerical instability in dynamical matrix.")

        dynevals, dynevecs_sam = np.linalg.eig(dyn_mat)
        self._logger.info(
            f"dynevecs_sam: {np.array2string(dynevecs_sam, precision=6, suppress_small=True)}\n"
        )
        self._logger.info(
            f"dynevals: {np.array2string(dynevals, precision=6, suppress_small=True)}\n"
        )
        self._logger.info(f"Eigenvalues: {dynevals}\n")
        self._logger.info(f"Absolute eigenvalues: {np.abs(dynevals)}\n")

        eV_to_J = 1.602177e-19
        ang_to_m = 1.0e-10
        AMU_to_kg = 1.66053e-27
        c = 2.9979458e10  # speed of light

        freq_thz = (
            np.sign(dynevals)
            * np.sqrt(np.abs(dynevals) * eV_to_J / (ang_to_m**2 * AMU_to_kg))
            * 1.0e-12
        )
        fc_eval = np.sign(fc_evals) * np.sqrt(np.abs(fc_evals))
        self._logger.info(f"Frequency in THz: {freq_thz}\n")
        idx_dyn = np.flip(np.argsort(freq_thz)[::-1])
        freq_thz = freq_thz[idx_dyn] / (2 * np.pi)
        dynevecs_sam = dynevecs_sam[:, idx_dyn]
        freq_cm = freq_thz * 1.0e12 / c
        self._logger.info(f"Frequency in wavenumbers: {freq_cm}\n")
        self.freq_cm = freq_cm

        self.dyn_freqs = [[freq_thz[i], freq_cm[i]] for i in range(self.num_sam)]
        self.fc_evals = fc_eval[idx_dyn]

        dynevecs = np.zeros(
            (self.num_sam, self.abinit_file.vars["natom"], 3), dtype=np.float64
        )
        for evec in range(self.num_sam):
            real_dynevec = np.zeros(
                (self.abinit_file.vars["natom"], 3), dtype=np.float64
            )
            for s in range(self.num_sam):
                real_dynevec += dynevecs_sam[s, evec] * self.dist_mat[s, :, :]
            dynevecs[evec, :, :] = real_dynevec

        self._logger.info(f"Dynevecs:\n{dynevecs} \n")

        mass_col = np.zeros((self.abinit_file.vars["natom"], 3), dtype=np.float64)
        atomind = 0
        for atype in range(self.abinit_file.vars["ntypat"]):
            for j in range(self.type_count[atype]):
                mass_col[atomind, :] = np.sqrt(self.mass_list[atype])
                atomind += 1

        phon_disp_eigs = np.zeros(
            (self.num_sam, self.abinit_file.vars["natom"], 3), dtype=np.float64
        )
        redmass_vec = np.zeros((self.abinit_file.vars["natom"], 1), dtype=np.float64)
        for mode in range(self.num_sam):
            phon_disp_eigs[mode, :, :] = np.divide(dynevecs[mode, :, :], mass_col)
            mag_squared = np.sum(phon_disp_eigs[mode, :, :] ** 2)
            redmass_vec[mode] = 1.0 / mag_squared
            phon_disp_eigs[mode, :, :] /= np.sqrt(mag_squared)

        self.phonon_vecs = phon_disp_eigs.astype(np.float64)
        self.red_mass = redmass_vec.astype(np.float64)
        self._logger.info(
            f"Reduced mass vector:\n{np.array2string(self.red_mass, precision=6, suppress_small=True)} \n"
        )
        self._logger.info(
            "Computation completed. Results stored in object attributes. \n"
        )

    def _imaginary_frequencies(self):
        negative_indices = []
        self._logger.info(
            f"Phonon vectors:\n{np.array2string(self.phonon_vecs, precision=6, suppress_small=True)} \n"
        )
        for index, fc_eval in enumerate(self.freq_cm):
            if fc_eval < self.unstable_threshold:
                negative_indices.append(index)
        self._logger.info(f"Unstable indices:\n{negative_indices} \n")
        return negative_indices if negative_indices else False

    def run_smodes(self, smodes_input):
        if not Path(settings.SMODES_PATH).is_file():
            raise FileNotFoundError(
                f"SMODES executable not found at: {settings.SMODES_PATH}"
            )
        command = f"{settings.SMODES_PATH} < {smodes_input} > output.log"
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"SMODES execution failed: {process.stderr}")
        return process.stdout

    def unstable_phonons(self):
        unstable_normalized = []
        unstable = self._imaginary_frequencies()
        if unstable is False:
            self._logger.info("No unstable phonons present")
            return False
        else:
            for i in unstable:
                flattened = self.phonon_vecs[i].flatten()
                norm_val = np.linalg.norm(flattened)
                normalized = flattened / norm_val
                normalized_matrix = normalized.reshape(self.phonon_vecs[i].shape)
                unstable_normalized.append(normalized_matrix)
        for i, mat in enumerate(unstable_normalized):
            self._logger.info(
                f"Matrix {i}:\n{np.array2string(mat, precision=4, suppress_small=True)}\n"
            )
            return unstable_normalized

    def symmadapt(self):
        self._loop_modes()
        self._perform_calculations()
        return self.unstable_phonons()

    def find_irrep(self):
        self._loop_modes()
        self._perform_calculations()
