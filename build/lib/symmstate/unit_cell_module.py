import numpy as np
import os
import re
import shutil
import sys
from pathlib import Path
from pymatgen.core import Structure, Lattice, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import warnings
import subprocess
from symmstate import SymmStateCore


class UnitCell(SymmStateCore):
    """
    Defines the UnitCell class which contains all the necessary information of a UnitCell.

    Initialization:
      - Directly input acell (array), rprim (ndarray), coordinates (ndarray), etc.
      - Use an abi_file (str)
      - Use a symmetry adapted basis

    Public Methods:
      - find_space_group(): Returns space group of the UnitCell
      - grab_reduced_coordinates(): Returns the reduced coordinates of the UnitCell
      - grab_cartesian_coordinates(): Returns the cartesian coordinates of the UnitCell
    """

    def __init__(
        self,
        acell=None,
        rprim=None,
        coordinates=None,
        coords_are_cartesian=False,
        elements=None,
        abi_file=None,
        smodes_file=None,
        target_irrep=None,
        symm_prec=1e-5,
        structure=None,
    ):
        """
        Initialize the class. Either provide the variables directly or specify an abifile for automatic configuration.

        Parameters:
        - acell: array-like, lattice constants
        - rprim: ndarray, primitive vectors
        - coordinates: ndarray, atomic positions
        - coords_are_cartesian: bool, if coordinates are in cartesian
        - elements: list, chemical elements
        - abi_file: str, path to Abinit file
        - smodes_file: str, path to SMODES file
        - target_irrep: str, target irrep for symmetry
        - symm_prec: float, symmetry precision
        - structure: pymatgen.core.Structure instance
        """

        # Initialization logic
        if structure is None:
            # Handle file-based initialization
            if abi_file:
                if os.path.isfile(abi_file):
                    self.abi_file = abi_file
                    acell, rprim, coordinates, coords_are_cartesian, elements = (
                        self._initialize_from_abifile(abi_file=abi_file)
                    )
                else:
                    raise FileExistsError(f"The abi file, {abi_file}, does not exist")
            elif smodes_file and target_irrep:
                if os.path.isfile(smodes_file):
                    self.smodes_file = smodes_file
                    params, _ = self._symmatry_adapted_basis(
                        smodes_file, target_irrep, symm_prec
                    )
                    acell, rprim, coordinates, coords_are_cartesian, elements = params
                else:
                    raise FileExistsError(
                        f"The smodes file, {smodes_file}, does not exist"
                    )
            else:
                # Direct input initialization
                required_fields = {
                    "acell": acell,
                    "rprim": rprim,
                    "coordinates": coordinates,
                    "elements": elements,
                }
                missing_fields = [
                    field_name
                    for field_name, value in required_fields.items()
                    if value is None
                ]
                if missing_fields:
                    raise ValueError(
                        f"Missing required fields: {', '.join(missing_fields)}"
                    )

                acell = np.array(acell, dtype=float)
                rprim = np.array(rprim, dtype=float)
                coordinates = np.array(coordinates, dtype=float)
                elements = np.array(elements, dtype=str)

            # Create pymatgen structure
            acell_vector = np.array(acell)
            rprim_scaled = rprim * acell_vector.T
            lattice = Lattice(rprim_scaled)
            self.structure = Structure(
                lattice=lattice,
                species=elements,
                coords=coordinates,
                coords_are_cartesian=coords_are_cartesian,
            )
        else:
            if isinstance(structure, Structure):
                self.structure = structure
            else:
                raise TypeError(
                    "Structure needs to be an instance of the pymatgen Structure class"
                )

        self.pseudopotentials = []
        self.ixc = None

    @staticmethod
    def _initialize_from_abifile(abi_file):
        """Extracts initialization variables from Abinit File"""

        temp_filepath = abi_file + ".temp"
        shutil.copy(abi_file, temp_filepath)

        with open(temp_filepath, "r") as f:
            lines = f.readlines()

        f.close()

        # Extract acell
        acell = []
        for i, line in enumerate(lines):
            if line.strip().startswith("acell"):
                # Extract different features of the acell feature and map it to a float where it will be inserted into a list.
                match = re.search(
                    r"(\d+)\*([-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*)", line
                )
                if match:
                    count = int(match.group(1))
                    value = float(match.group(2))
                    acell = [value] * count
                else:
                    acell = list(
                        map(
                            float,
                            re.findall(r"[-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*", line),
                        )
                    )
                # Delete extracted lines in copy
                del lines[i]

        if not acell:  # Check if acell is still an empty list
            raise Exception("acell is missing in the Abinit file!")

        # Extract primitive vectors
        rprim = []
        for i, line in enumerate(lines):
            if line.strip() == "rprim":
                del lines[i]
                j = i
                while j < len(lines) and re.match(
                    r"^\s*[-+]?[0-9]*\.?[0-9]+", lines[j]
                ):
                    rprim.append(list(map(float, lines[j].split())))
                    del lines[j]
                break
        else:
            # Default rprim value if not specified
            rprim = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        rprim = np.array(rprim)

        # Extract coordinates (xred or cartesian)
        coord_type = None
        coordinates = []
        for i, line in enumerate(lines):
            if line.strip() == "xred":
                coord_type = "reduced"
                del lines[i]
                j = i
                while j < len(lines) and re.match(
                    r"^\s*[-+]?[0-9]*\.?[0-9]+", lines[j]
                ):
                    coordinates.append(list(map(float, lines[j].split())))
                    del lines[j]
                break
            elif line.strip() == "xcart":
                coord_type = "cartesian"
                del lines[i]
                j = i
                while j < len(lines) and re.match(
                    r"^\s*[-+]?[0-9]*\.?[0-9]+", lines[j]
                ):
                    coordinates.append(list(map(float, lines[j].split())))
                    del lines[j]
                break

        coordinates = np.array(coordinates)
        if len(coordinates) == 0:  # Check if coordinates list is empty
            raise Exception("coordinates are missing in the Abinit file!")

        coords_are_cartesian = False
        if coord_type == "xcart":
            coords_are_cartesian = True

        # Extract natom
        num_atoms = None
        for i, line in enumerate(lines):
            if line.strip().startswith("natom"):
                match = re.search(r"\d+", line)
                if match:
                    num_atoms = int(match.group())
                del lines[i]
                break

        if num_atoms is None:
            raise Exception("natom is missing in the Abinit file!")

        # Extract znucl
        znucl = []
        for i, line in enumerate(lines):
            if line.strip().startswith("znucl"):
                znucl = list(
                    map(int, re.findall(r"\d+", line))
                )  # Fixed typo in re.finall to re.findall
                del lines[i]
                break

        if not znucl:  # Check if znucl is still an empty list
            raise Exception("znucl is missing in the Abinit file!")

        # Extract typat
        typat = []
        for i, line in enumerate(lines):
            if line.strip().startswith("typat"):
                # TODO: Length may be hardcoded
                typat_tokens = re.findall(r"(\d+\*\d+|\d+)", line)
                for token in typat_tokens:
                    if "*" in token:
                        count, value = map(int, token.split("*"))
                        typat.extend([value] * count)
                    else:
                        typat.append(int(token))
                del lines[i]
                break

        if not typat:  # Check if typat is still an empty list
            raise Exception("typat is missing in the Abinit file!")

        # Convert znucl to element symbols
        elements_symbols = [Element.from_Z(Z).symbol for Z in znucl]

        # Use typat to create the element list
        elements = [elements_symbols[i - 1] for i in typat]

        return acell, rprim, coordinates, coords_are_cartesian, elements

    @staticmethod
    def _symmatry_adapted_basis(
        smodes_file, target_irrep, symm_prec, smodes_path="../isobyu/smodes"
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
        command = f"{smodes_path} < {smodes_file}"
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

        for l in range(8, len(target_output)):
            line_content = target_output[l].split()
            if target_output[l] == "Symmetry modes:":
                break
            if len(line_content) >= 4:
                atom_names.append(line_content[1])
                atom_positions_raw.append(
                    [
                        float(line_content[2]),
                        float(line_content[3]),
                        float(line_content[4]),
                    ]
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

        clean_list = UnitCell._generate_clean_list()
        shape_cell = UnitCell._clean_matrix(shape_cell, clean_list, symm_prec=symm_prec)

        prec_lat_array = np.array([prec_lat_param, prec_lat_param, prec_lat_param])

        # Primitive vectors definition
        rprim = np.multiply(shape_cell, prec_lat_array)

        atom_positions = UnitCell._clean_positions(
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
        dist_mat, sam_atom_label = UnitCell._calculate_displacement_matrix(
            target_output, num_modes, num_atoms, start_line
        )

        dist_mat = UnitCell._orthogonalize_sams(dist_mat, num_modes, num_atoms)

        crossDot = np.dot(np.cross(rprim[0, :], rprim[1, :]), np.transpose(rprim[2, :]))
        crossdot_ispos = crossDot > 0

        # TODO: Do something if this is not positive
        if crossdot_ispos == False:
            warnings.warn("Abinit requires this to be positive!")

        coords_are_cartesian = False

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

    def grab_reduced_coordinates(self):
        """Grabs the reduced coordinates of the UnitCell"""
        return np.array(self.structure.frac_coords)

    def grab_cartesian_coordinates(self):
        """Grabs the cartesian coordinates of the UnitCell"""
        return np.array(self.structure.cart_coords)

    def find_space_group(self):
        """Calculates and returns the space group of the unit cell."""
        analyzer = SpacegroupAnalyzer(self.structure)
        return (analyzer.get_space_group_number(), analyzer.get_space_group_symbol())

    def perturbations(self, perturbation, coords_are_cartesian=False):
        """
        Apply a given perturbation to the unit cell coordinates and return a new UnitCell.

        Args:
            perturbation (np.ndarray): A numpy array representing the perturbation to be applied.
            coords_are_cartesian (bool): If True, treats perturbation as cartesian, else reduced.

        Returns:
            UnitCell: A new instance of UnitCell with perturbed coordinates.
        """

        # Ensure the perturbation has the correct shape
        perturbation = np.array(perturbation, dtype=float)
        if perturbation.shape != self.structure.frac_coords.shape:
            raise ValueError(
                "Perturbation must have the same shape as the fractional coordinates."
            )

        # Calculate new fractional coordinates by adding the perturbation
        if coords_are_cartesian:
            new_frac_coords = self.structure.cart_coords + perturbation
        else:
            new_frac_coords = self.structure.frac_coords + perturbation

        # Create a new Structure object with the updated fractional coordinates
        perturbed_structure = Structure(
            lattice=self.structure.lattice,
            species=self.structure.species,
            coords=new_frac_coords,
            coords_are_cartesian=coords_are_cartesian,  # Ensure coordinates are treated as fractional
        )

        # Return a new instance of UnitCell with the perturbated structure
        return UnitCell(structure=perturbed_structure)

    def change_coordinates(self, new_coordinates, coords_are_cartesian=False):
        """
        Updates the coordinates of the unit cell to new values and resets the energy attribute.

        Args:
            new_coordinates (np.ndarray): New array of coordinates to set for the unit cell.
            coords_are_cartesian (bool): If True, indicates the new coordinates are in Cartesian form.
        """
        self.structure = Structure(
            lattice=self.structure.lattice,
            species=self.structure.species,
            coords=new_coordinates,
            coords_are_cartesian=coords_are_cartesian,
        )

    def upload_pseudopotentials(self, *files, dest_folder_name="pseudopotentials"):
        """
        Moves files specified in the arguments to the designated pseudopotentials folder.

        Parameters:
        - *files: A variable number of file paths to be moved.
        - dest_folder_name (str): Destination folder name where files will be moved. Default is 'pseudopotentials'.
        """
        relative_path =  SymmStateCore.upload_files_to_package(*files, dest_folder_name=dest_folder_name)
        return relative_path
    


    def __repr__(self):
        """
        Using Pymatgen's native repr to represent the UnitCell
        """
        return print(self.structure)
