from symmstate.unit_cell_module import UnitCell
import numpy as np
import os
import re
import shutil
import sys
import tempfile
import copy
from pathlib import Path
import warnings
from pymatgen.core import Element

# To ensure the file is calling from the correct directory.
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))


class AbinitUnitCell(UnitCell):
    """
    Defines the AbinitUnitCell class, which provides functionality for Abinit simulations, extending `UnitCell`.

    Attributes:
        abi_file (str): Path to the main Abinit file containing unit cell info.
        file_name (str): Derived from `abi_file`.
        ecut (int): Energy cutoff value.
        ecutsm (float): Smearing on the energy cutoff.
        nshiftk (int): Number of shifts for k-points.
        shiftk (list[float]): Shift vectors for k-points.
        nstep (int): Steps for SCF calculation.
        diemac (float): Macroscopic dielectric constant.
        ixc (int): Exchange-correlation functional index.
        pp_dirpath (str): Directory for pseudopotentials.
        pseudos (str): Pseudopotential filenames.
        kptrlatt (list[list[int]]): Matrix defining reciprocal space vectors.
        nband (int): Number of bands for electronic calculations.
        toldfe (str): Total energy difference tolerance in calculations.
        convergence_file (str): File path with convergence parameters.
        batchScriptHeader_path (str): Batch script header file path.
        runningJobs (list[int]): Job IDs for currently running jobs.
        energy (float): Energy of the current configuration.

    Public Methods:
        findSpaceGroup(): Determines and returns the space group of the unit cell.
        convertToXcart(): Converts and returns Cartesian coordinates of the unit cell.
        convertToXred(): Converts and returns reduced coordinates of the unit cell.
        write_ground_workfunction_file(output_path): Creates an Abinit file for calculating work function.
        write_phonon_dispersion_file(output_path): Creates an Abinit file for phonon dispersion calculation.
        write_custom_abifile(output_file, header_file, toldfe=True): Writes a custom Abinit .abi file.
        all_jobs_finished(): Checks if all submitted jobs have finished.
        wait_for_jobs_to_finish(check_time=60): Waits until all submitted jobs are completed.
        run_abinit(input_file='abinit', batch_name='abinit_job', ...): Runs Abinit with specified settings.
        write_batch_script(batch_script_header_file='default_batch_file', ...): Writes a batch script for running simulations.
        perturbations(pert): Applies perturbations to unit cell coordinates and returns a new instance.
        copy_abinit_unit_cell(): Creates a deep copy of the current instance.
        run_energy_calculation(host_spec='mpirun -hosts=localhost -np 30'): Executes an energy calculation for the unit cell.
        grab_energy(abo_file=None): Retrieves the total energy from an Abinit output file.
        change_coordinates(new_coordinates, cartesian=False, reduced=False): Updates the coordinates of the unit cell.
    """

    # TODO: Make a method that will place all pseudopotentials into the pseudopotential folder in the program.

    def __init__(
        self,
        abi_file=None,
        convergence_file=None,
        smodes_input=None,
        target_irrep=None,
        symmetry_informed_basis=False,
    ):
        """
        Initializes an instance of AbinitUnitCell.

        Args:
            abi_file (str): Path to the Abinit input file containing unit cell details.
            convergence_file (str, optional): Path to a file with convergence parameters. Defaults to None.
            batch_script_header_file (str, optional): Path to a batch script header file. Defaults to None.
        """
        # Call the parent class's initializer with the keyword arguments
        self.abi_file = str(abi_file)

        # Allows the user to still take convergence variables from abi file
        if symmetry_informed_basis:
            super().__init__(smodes_file=smodes_input, target_irrep=target_irrep)
        else:
            super().__init__(abi_file=abi_file)
        self.natom, self.ntypat, self.typat, self.znucl = AbinitUnitCell._process_atoms(
            self.structure.species
        )
        self.rprim = np.array(self.structure.lattice.matrix)
        self.coordinates_xred = self.structure.frac_coords
        self.coordinates_xcart = self.structure.cart_coords

        a, b, c = self.structure.lattice.abc
        self.acell = [a, b, c]

        # Convergence attributes
        self.ecut = None
        self.ecutsm = None
        self.nshiftk = None
        self.shiftk = []
        self.nstep = None
        self.diemac = None
        self.kptrlatt = None
        self.nband = None

        # Only one can be defined
        self.toldfe = None
        self.tolvrs = None
        self.tolsym = None

        # If initialized correctly, will contain choice of conv_criteria. Will be tuple
        self.conv_criteria = None

        # Initialize additional attributes specific to AbinitUnitCell
        if convergence_file is None:
            self._initialize_convergence_from_file()

        self.convergence_file = convergence_file

        # -----------------------------
        # Abinit Specific Calculations
        # -----------------------------

        # Energy of the unit cell
        self.energy = None

        # Electric properties of the cell
        self.piezo_tensor_clamped = None
        self.piezo_tensor_relaxed = None
        self.flexo_tensor = None

    # --------------------------
    # Initialization Methods
    # --------------------------

    @staticmethod
    def _process_atoms(atom_list):
        # Calculate the total number of atoms
        num_atoms = len(atom_list)

        # Get the unique elements and their respective indices
        unique_elements = list(dict.fromkeys(atom_list))
        element_index = {element: i + 1 for i, element in enumerate(unique_elements)}

        # Create typat list based on unique elements' indices
        typat = [element_index[element] for element in atom_list]

        # Create znucl list with atomic numbers using pymatgen
        znucl = [Element(el).Z for el in unique_elements]

        return num_atoms, len(unique_elements), typat, znucl

    def _initialize_convergence_from_file(self):
        """
        Extracts convergence features from the Abinit file by parsing the required parameters.
        Raises exceptions if critical parameters are missing from the file.
        """

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Copy the contents of the original file to the temporary file
            shutil.copyfile(self.abi_file, temp_file.name)

            # Open the temporary file for reading
            with open(temp_file.name, "r") as f:
                lines = f.readlines()

        # Extract ecut
        for i, line in enumerate(lines):
            if line.strip().startswith("ecut"):
                match = re.search(r"\d+", line)
                if match:
                    ecut = int(match.group())
                del lines[i]
                break

        self.ecut = ecut
        if self.ecut is None:
            raise Exception("ecut is missing in the Abinit file!")

        # Extract ecutsm
        for i, line in enumerate(lines):
            if line.strip().startswith("ecutsm"):
                match = re.search(r"\d+\.\d+|\d+", line)
                if match:
                    ecustsm = float(match.group())
                del lines[i]
                break

        self.ecutsm = ecustsm
        if ecustsm is None:
            # Default value of Abinit
            self.ecutsm = 0.5

        # Extract nshiftk
        for i, line in enumerate(lines):
            if line.strip().startswith("nshiftk"):
                match = re.search(r"\d+", line)
                if match:
                    nshiftk = int(match.group())
                del lines[i]
                break

        self.nshiftk = nshiftk
        if self.nshiftk is None:
            raise Exception("nshiftk is missing in the Abinit file!")

        # Extract shiftk
        for i, line in enumerate(lines):
            if line.strip().startswith("shiftk"):
                # Extract different features of the acell feature and map it to a float where it will be inserted into a list.
                match = re.search(
                    r"(\d+)\*([-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*)", line
                )
                if match:
                    count = int(match.group(1))
                    value = float(match.group(2))
                    shiftk = [value] * count
                else:
                    shiftk = list(
                        map(
                            float,
                            re.findall(r"[-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*", line),
                        )
                    )
                # Delete extracted lines in copy
                del lines[i]

        self.shiftk = shiftk
        if self.shiftk is None:
            raise Exception("shiftk is missing in the Abinit file!")

        # Extract nband
        for i, line in enumerate(lines):
            if line.strip().startswith("nband"):
                match = re.search(r"\d+", line)
                if match:
                    nband = int(match.group())
                del lines[i]
                break

        self.nband = nband
        if self.nband is None:
            raise Exception("nband is missing in the Abinit file!")

        # Extract nstep
        for i, line in enumerate(lines):
            if line.strip().startswith("nstep"):
                match = re.search(r"\d+", line)
                if match:
                    nstep = int(match.group())
                del lines[i]
                break

        self.nstep = nstep
        if self.nstep is None:
            # Default Abinit value
            self.nstep = 30

        # Extract diemac
        for i, line in enumerate(lines):
            if line.strip().startswith("diemac"):
                match = re.search(r"\d+\.\d+|\d+", line)
                if match:
                    diemac = float(match.group())
                del lines[i]
                break

        self.diemac = diemac
        if self.diemac is None:
            # Default Abinit value
            self.diemac = 4.0

        # Extract toldfe or tolvrs or tolsym
        for i, line in enumerate(lines):
            if line.strip().startswith("toldfe"):
                match = re.search(r"toldfe\s+(\d+\.\d+d[+-]?\d+)", line)
                if match:
                    toldfe = match.group(1)
                del lines[i]
                self.toldfe = toldfe
                break
            elif line.strip().startswith("tolvrs"):
                match = re.search(r"tolvrs\s+(\d+\.\d+d[+-]?\d+)", line)
                if match:
                    tolvrs = match.group(1)
                del lines[i]
                self.tolvrs = tolvrs
                break
            elif line.strip().startswith("tolsym"):
                match = re.search(r"tolsym\s+(\d+\.\d+d[+-]?\d+)", line)
                if match:
                    tolsym = match.group(1)
                del lines[i]
                self.tolsym = tolsym
                break

        conv_criteria = {
            "tolvrs": self.tolvrs,
            "toldfe": self.toldfe,
            "tolsym": self.tolsym,
        }

        # Check that exactly one was declared
        non_none_attrs = {
            name: value for name, value in conv_criteria.items() if value is not None
        }

        # Check if exactly one attribute is not none
        if len(non_none_attrs) != 1:
            warnings.warn(
                "Abinit typically only accepts either tolsym or toldfe or tolvrs"
            )
        else:
            attr_name, attr_value = next(iter(non_none_attrs.items()))
            self.conv_criteria = (attr_name, attr_value)

        # Extract ixc
        for i, line in enumerate(lines):
            if line.strip().startswith("ixc"):
                match = re.search(r"[-+]?\d+", line)
                if match:
                    ixc = int(match.group())
                del lines[i]
                break

        self.ixc = ixc
        if self.ixc is None:
            raise Exception("ixc is missing in the Abinit file!")

        # ---------------------------------------------
        # Pseudopotential extraction and intialization
        # ---------------------------------------------

        # Extract pp_dirpath
        for i, line in enumerate(lines):
            if line.strip().startswith("pp_dirpath"):
                match = re.search(r'pp_dirpath\s+"([^"]+)"', line)
                if match:
                    pp_dirpath = str(match.group(1))
                del lines[i]
                break

        if pp_dirpath is None:
            raise Exception("pp_dirpath is missing in the Abinit file!")

        # Extract pseudos
        for i, line in enumerate(lines):
            if line.strip().startswith("pseudos"):
                match = re.search(r'pseudos\s+"([^"]+)"', line)
                if match:
                    pseudos = str(match.group(1))
                del lines[i]
                break

        pseudos = pseudos.split()

        # Take list of pseudopotential files and upload them into Abinit folder
        for pseudo in pseudos:
            file_path = str(os.path.join(pp_dirpath, pseudo))

            self.upload_pseudopotentials(file_path)

            # Add pseudopotential as an attribute of the Unit Cell
            self.pseudopotentials.append(str(pseudo))

        # Extract kptrlatt
        kptrlatt = []
        for i, line in enumerate(lines):
            if line.strip() == "kptrlatt":
                del lines[i]
                j = i
                while j < len(lines) and re.match(r"^\s*[-+]?\d+", lines[j]):
                    kptrlatt.append(list(map(int, lines[j].split())))
                    del lines[j]
                break

        self.kptrlatt = kptrlatt

        os.remove(temp_file.name)

    # --------------------------
    # Utilities
    # --------------------------

    def copy_abinit_unit_cell(self):
        """
        Creates a deep copy of the current AbinitUnitCell instance.

        Returns:
            AbinitUnitCell: A new instance that is a deep copy of the current instance.
        """
        # Perform a deep copy to ensure all nested objects are also copied
        copied_cell = copy.deepcopy(self)
        return copied_cell

    def change_coordinates(self, new_coordinates, coords_are_cartesian=False):
        # Update structure
        super().change_coordinates(new_coordinates, coords_are_cartesian)

        # Update class attributes
        self.coordinates_xcart = self.structure.cart_coords
        self.coordinates_xred = self.structure.frac_coords

    def perturbations(self, perturbation, coords_is_cartesian=False):
        """
        Applies a given perturbation to the unit cell's coordinates and returns a new AbinitUnitCell object.

        Args:
            pert (np.ndarray): Array representing the perturbation to be applied to current coordinates.

        Returns:
            AbinitUnitCell: A new instance of UnitCell with updated (perturbed) coordinates.
        """

        # Ensure the perturbation has the correct shape
        perturbation = np.array(perturbation, dtype=float)
        if perturbation.shape != self.coordinates_xred.shape:
            raise ValueError(
                "Perturbation must have the same shape as the coordinates."
            )

        if coords_is_cartesian:
            new_coordinates = self.coordinates_xcart + perturbation
        else:
            new_coordinates = self.coordinates_xred + perturbation

        copy_cell = self.copy_abinit_unit_cell()
        # Calculate new coordinates by adding the perturbation
        copy_cell.change_coordinates(
            new_coordinates=new_coordinates, coords_are_cartesian=coords_is_cartesian
        )

        return copy_cell
