import numpy as np
from pathlib import Path
import subprocess
import os
from pymatgen.core import Structure, Lattice, Element
from symmstate.config.symm_state_settings import settings
import warnings


class SymmAdaptedBasis:
    def __init__(self):
        pass

    @staticmethod
    def symmatry_adapted_basis(
        smodes_file,
        target_irrep,
        symm_prec=1.0e-5,
    ):
        """
        Extract header information from SMODES input file and store it in class attributes.

        Args:
            smodes_input (str): Path to the SMODES input file.

        Raises:
            FileNotFoundError: If the SMODES executable is not found.

        Returns:
            List of initialization parameters and extracted data.
        """

        if not Path(smodes_file).is_file():
            raise FileNotFoundError(
                f"SMODES executable not found at: {smodes_file}. Current directory is {os.getcwd()}"
            )

        # Open and read SMODES input file
        with open(smodes_file) as s:
            s_lines = s.readlines()

        # Parse lattice parameters
        prec_lat_param = [float(x) for x in s_lines[0].split()]

        print(f"Precision Lattice Parameters:\n {prec_lat_param}\n ")
        acell = [1, 1, 1]

        # Execute SMODES and process output
        command = f"{str(settings.SMODES_PATH)} < {smodes_file}"
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        proc.wait()
        output = proc.stdout.read().decode("ascii")

        print(f"Printing smodes output: \n \n {output} \n")

        proc.stdout.close()

        # Process the output from SMODES
        start_target = 999
        end_target = 0
        outlist = output.split("\n")

        for line in range(len(outlist)):
            line_content = outlist[line].split()
            if (
                len(line_content) > 1
                and line_content[0] == "Irrep"
                and line_content[1] == target_irrep
            ):
                start_target = line
            if len(line_content) > 0 and start_target < 999:
                if line_content[0] == "***********************************************":
                    end_target = line
                    break

        target_output = outlist[start_target:end_target]
        israman = False
        transmodes = None
        isir = False
        if target_output[3].split()[0] == "These":
            transmodes = True
            del target_output[3]

        if target_output[3].split()[0] == "IR":
            isir = True
            del target_output[3]

        if target_output[3].split()[0] == "Raman":
            israman = True
            del target_output[3]

        # Parse degeneracy and number of modes
        degeneracy = int(target_output[1].split()[-1])
        num_modes_without_degen = int(target_output[2].split()[-1])
        num_modes = num_modes_without_degen // degeneracy

        print(f"Degeneracy: {degeneracy}\n")
        print(f"Number of Modes: {num_modes_without_degen}")
        print(f"(Meaning {num_modes} modes to find) \n")

        # Process lattice vectors and atomic positions
        v1 = [float(i) for i in target_output[4].split()]
        v2 = [float(i) for i in target_output[5].split()]
        v3 = [float(i) for i in target_output[6].split()]
        shape_cell = np.array([v1, v2, v3])

        atom_names = []
        atom_positions_raw = []

        # Loop through the target output starting from line 8.
        for l in range(8, len(target_output)):
            line = target_output[l].strip()
            # Use a more robust termination condition.
            if "Symmetry modes:" in line:
                break
            tokens = line.split()
            # Expect exactly 5 tokens: [index, atom name, x, y, z]
            if len(tokens) < 5:
                continue  # skip lines that do not have enough tokens
            atom_names.append(tokens[1])
            try:
                pos = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
            except ValueError:
                continue  # skip lines where coordinate conversion fails
            atom_positions_raw.append(pos)

        # Check that the two lists have equal length; otherwise raise an error.
        if len(atom_names) != len(atom_positions_raw):
            raise ValueError(
                f"Parsing error: {len(atom_names)} atom names but {len(atom_positions_raw)} positions were extracted. Check your SMODES file format."
            )

        # Number of atoms
        num_atoms = len(atom_names)

        # Dictionary to count occurrences of each element
        count_dict = {}

        # Iterate over each element in the input array
        for item in atom_names:
            if item in count_dict:
                count_dict[item] += 1
            else:
                count_dict[item] = 1

        multiplicity_list = []
        seen = set()

        for item in atom_names:
            if item not in seen:
                multiplicity_list.append(count_dict[item])
                seen.add(item)

        type_count = multiplicity_list

        result = [
            (index + 1)
            for index, value in enumerate(multiplicity_list)
            for _ in range(value)
        ]
        typat = result

        clean_list = SymmAdaptedBasis._generate_clean_list()
        shape_cell = SymmAdaptedBasis._clean_matrix(
            shape_cell, clean_list, symm_prec=symm_prec
        )

        prec_lat_array = np.array([prec_lat_param, prec_lat_param, prec_lat_param])

        # Primitive vectors definition
        rprim = np.multiply(shape_cell, prec_lat_array)

        atom_positions = SymmAdaptedBasis._clean_positions(
            atom_positions_raw, prec_lat_param, clean_list, symm_prec=symm_prec
        )
        print(f"Smodes Unit Cell Coordinates:\n {atom_positions} \n")
        coordinates = atom_positions
        pos_mat_cart = coordinates.copy()

        atom_names_nodup = list(dict.fromkeys(atom_names))
        type_list = atom_names_nodup

        # Get atomic details using pymatgen's Element class
        atomic_num_list = [Element(name).Z for name in atom_names_nodup]
        atomic_mass_list = [Element(name).atomic_mass for name in atom_names_nodup]

        znucl = atomic_num_list

        # Symmetry adapted related attributes
        num_sam = num_modes
        mass_list = atomic_mass_list
        pos_mat_cart = pos_mat_cart

        start_line = num_atoms + 11
        dist_mat, sam_atom_label = SymmAdaptedBasis._calculate_displacement_matrix(
            target_output, num_modes, num_atoms, start_line
        )

        dist_mat = SymmAdaptedBasis._orthogonalize_sams(dist_mat, num_modes, num_atoms)

        crossDot = np.dot(np.cross(rprim[0, :], rprim[1, :]), np.transpose(rprim[2, :]))
        crossdot_ispos = crossDot > 0

        # TODO: Do something if this is not positive
        if crossdot_ispos == False:
            warnings.warn("Abinit requires this to be positive!")

        # TODO: Do not assume that the coordinates will be reduced.
        coords_are_cartesian = True

        # Convert znucl to element symbols
        elements_symbols = [Element.from_Z(Z).symbol for Z in znucl]

        # Use typat to create the element list
        elements = [elements_symbols[i - 1] for i in typat]

        return [acell, rprim, coordinates, coords_are_cartesian, elements], [
            transmodes,
            isir,
            israman,
            type_count,
            type_list,
            num_sam,
            mass_list,
            pos_mat_cart,
            dist_mat,
            sam_atom_label,
        ]

    @staticmethod
    def _generate_clean_list():
        """
        Generate a list of rational approximations for cleanup.

        Returns:
            list: List of clean values for matrix approximation.
        """
        clean_list = [1.0 / 3.0, 2.0 / 3.0]
        for i in range(1, 10):
            for base in [np.sqrt(3), np.sqrt(2)]:
                clean_list.extend(
                    [
                        base / float(i),
                        2 * base / float(i),
                        3 * base / float(i),
                        4 * base / float(i),
                        5 * base / float(i),
                        float(i) / 6.0,
                        float(i) / 8.0,
                    ]
                )
        return clean_list

    @staticmethod
    def _clean_matrix(matrix, clean_list, symm_prec):
        """
        Clean a matrix by replacing approximate values with exact ones using clean_list.

        Args:
            matrix (np.ndarray): Input matrix to be cleaned.
            clean_list (list): List of target values for cleaning.
            symm_prec (float): Precision for symmetry operations.

        Returns:
            np.ndarray: Cleaned matrix.
        """
        for n in range(matrix.shape[0]):
            for i in range(matrix.shape[1]):
                for c in clean_list:
                    if abs(abs(matrix[n, i]) - abs(c)) < symm_prec:
                        matrix[n, i] = np.sign(matrix[n, i]) * c
        return matrix

    @staticmethod
    def _clean_positions(positions, prec_lat_param, clean_list, symm_prec):
        """
        Clean atomic positions and convert using lattice parameters.

        Args:
            positions (list): List of raw atomic positions.
            prec_lat_param (list): Lattice parameters for conversion.
            clean_list (list): List of values to use for cleaning.
            symm_prec (float): Precision for symmetry operations.

        Returns:
            np.ndarray: Cleaned and converted atomic positions.
        """
        # Copy positions to avoid modifying the original input data
        cleaned_positions = positions.copy()

        for n, pos in enumerate(cleaned_positions):
            for i in range(3):
                for c in clean_list:
                    if abs(abs(pos[i]) - abs(c)) < symm_prec:
                        pos[i] = np.sign(pos[i]) * c
                pos[i] *= prec_lat_param[i]

        # Convert to a NumPy array to ensure consistent processing
        cleaned_positions = np.array(cleaned_positions)

        # Ensure dimensions are correct
        if cleaned_positions.ndim != 2 or cleaned_positions.shape[1] != 3:
            raise ValueError(
                f"Cleaned positions do not have expected shape (n_atoms, 3): {cleaned_positions.shape}"
            )

        return np.array(cleaned_positions)

    @staticmethod
    def _calculate_displacement_matrix(target_output, num_modes, num_atoms, start_line):
        """
        Calculate the initial displacement matrix from SMODES output.

        Args:
            target_output (list): Parsed output lines from SMODES execution.
            num_modes (int): Number of modes considered in calculations.
            num_atoms (int): Total number of atoms present.
            start_line (int): Line number in output where parsing begins.

        Returns:
            tuple: Calculated displacement matrix and SAM atom labels.
        """
        dist_mat = np.zeros((num_modes, num_atoms, 3))
        mode_index = -1
        sam_atom_label = [None] * num_modes

        for l in range(start_line, len(target_output)):
            line_content = target_output[l].split()
            if target_output[l] == "------------------------------------------":
                mode_index += 1

            else:
                atom = int(line_content[0]) - 1
                sam_atom_label[mode_index] = line_content[1]
                disp1, disp2, disp3 = map(float, line_content[2:5])
                dist_mat[mode_index, atom, 0] = disp1
                dist_mat[mode_index, atom, 1] = disp2
                dist_mat[mode_index, atom, 2] = disp3

        return dist_mat, sam_atom_label

    @staticmethod
    def _orthogonalize_sams(dist_mat, num_modes, num_atoms):
        """
        Normalize and orthogonalize the systematic atomic modes (SAMs).

        Args:
            dist_mat (np.ndarray): Initial displacement matrix.
            num_modes (int): Number of modes.
            num_atoms (int): Number of atoms.

        Returns:
            np.ndarray: Orthogonalized matrix of SAMs.
        """

        # Normalize the SAMs
        for m in range(0, num_modes):
            norm = np.linalg.norm(dist_mat[m, :, :])
            if norm == 0:
                raise ValueError(
                    f"Zero norm encountered at index {m} during normalization."
                )
            dist_mat[m, :, :] /= norm

        # Orthogonalize the SAMs using a stable Gram-Schmidt Process
        orth_mat = np.zeros((num_modes, num_atoms, 3))
        for m in range(0, num_modes):
            sam = dist_mat[m, :, :]

            for n in range(m):
                proj = np.sum(np.multiply(sam, orth_mat[n, :, :])) * orth_mat[n, :, :]
                sam -= proj

            # Re-normalize
            norm = np.linalg.norm(sam)
            if norm > 0:
                orth_mat[m, :, :] = sam / norm
            else:
                # Handle the zero norm case, e.g., assigning a zero matrix or handling it differently
                orth_mat[m, :, :] = np.zeros_like(sam)
                print(
                    f"Warning: Zero norm encountered at index {m} during orthogonalization."
                )

        return orth_mat
