import numpy as np
import os
import re
import shutil
import sys
from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# To ensure the file is calling from the correct directory. 
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))


class UnitCell:
    """
    Defines the UnitCell class which contains the all the necessary information of a UnitCell

    Initialization: 
        1.) Directly input acell (arr), rprim (np.array), coordinates (np.array), coord_type (str), num_atoms (int),
        znucl (arr), ntypat (int), typat (arr), header_path [optional] (str)
        2.) Use an abi_file (str)

    Public Methods: 
        find_spaceGroup(): Returns space group of the UnitCell
        xred(): Returns the reduced coordinates of the UnitCell
        xcart(): Returns the cartesian coordinates of the UnitCell

    """
    
    def __init__(self, acell=None, rprim=None, coordinates=None, coord_type=None, num_atoms=None, ntypat=None, znucl=None, typat=None, header_path=None, abi_file=None):
        """
        Initialize the class. Either provide the variables directly or specify an abifile for automatic configuration.

        Acceptable keyword arguments are:
        - acell, rprim, coordinates, coord_type, num_atoms, ntypat, znucl, typat, header_path, abi_file
        """

        self.header_path = header_path

        # If 'abi_file' is provided and exists, initialize from file
        if abi_file is not None:
            if os.path.isfile(abi_file):
                self.abi_file = abi_file
                self._initialize_from_abifile()
            else:
                raise FileExistsError("The abi file does not exist")
        else:
            # Use the provided keyword arguments
            required_fields = {
                'acell': acell,
                'rprim': rprim,
                'coordinates': coordinates,
                'coord_type': coord_type,
                'num_atoms': num_atoms,
                'atom_types': ntypat,
                'znucl': znucl,
                'typat': typat
            }
            missing_fields = [field_name for field_name, value in required_fields.items() if value is None]

            if missing_fields:
                raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

            self.acell = np.array(acell, dtype=float)
            self.rprim = np.array(rprim, dtype=float)
            self.coordinates = np.array(coordinates, dtype=float)
            self.coord_type = coord_type
            self.num_atoms = num_atoms
            self.typat = np.array(typat, dtype=int)
            self.ntypat = int(ntypat)
            self.znucl = np.array(znucl, dtype=int)

    def _initialize_from_abifile(self):
        """Extracts initialization variables from Abinit File"""

        temp_filepath = self.abi_file + ".temp"
        shutil.copy(self.abi_file, temp_filepath)

        with open(temp_filepath, 'r') as f: 
            lines = f.readlines()

        f.close()

        # Extract acell
        acell = []
        for i, line in enumerate(lines):
            if line.strip().startswith('acell'):
                # Extract different features of the acell feature and map it to a float where it will be inserted into a list.
                match = re.search(r"(\d+)\*([-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*)", line)
                if match:
                    count = int(match.group(1))
                    value = float(match.group(2))
                    acell = [value] * count
                else:
                    acell = list(map(float, re.findall(r"[-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*", line)))
                # Delete extracted lines in copy
                del lines[i]

        self.acell = acell
        if not acell:  # Check if acell is still an empty list
            raise Exception("acell is missing in the Abinit file!")

        # Extract primitive vectors
        rprim = []
        for i, line in enumerate(lines):
            if line.strip() == 'rprim':
                del lines[i]
                j = i
                while j < len(lines) and re.match(r"^\s*[-+]?[0-9]*\.?[0-9]+", lines[j]):
                    rprim.append(list(map(float, lines[j].split())))
                    del lines[j]
                break
        else:
            # Default rprim value if not specified
            rprim = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
            
        self.rprim = np.array(rprim)

        # Extract coordinates (xred or cartesian)
        coord_type = None
        coordinates = []
        for i, line in enumerate(lines):
            if line.strip() == 'xred':
                coord_type = 'reduced'
                del lines[i]
                j = i
                while j < len(lines) and re.match(r"^\s*[-+]?[0-9]*\.?[0-9]+", lines[j]):
                    coordinates.append(list(map(float, lines[j].split())))
                    del lines[j]
                break
            elif line.strip() == 'xcart':
                coord_type = 'cartesian'
                del lines[i]
                j = i
                while j < len(lines) and re.match(r"^\s*[-+]?[0-9]*\.?[0-9]+", lines[j]):
                    coordinates.append(list(map(float, lines[j].split())))
                    del lines[j]
                break

        self.coordinates = np.array(coordinates)
        self.coord_type = coord_type
        if not coordinates:  # Check if coordinates list is empty
            raise Exception("coordinates are missing in the Abinit file!")

        # Extract natom
        num_atoms = None
        for i, line in enumerate(lines):
            if line.strip().startswith('natom'):
                match = re.search(r"\d+", line)
                if match:
                    num_atoms = int(match.group())
                del lines[i]
                break

        self.num_atoms = num_atoms
        if num_atoms is None:
            raise Exception("natom is missing in the Abinit file!")

        # Extract ntypat
        ntypat = None
        for i, line in enumerate(lines):
            if line.strip().startswith('ntypat'):
                match = re.search(r"\d+", line)
                if match:
                    ntypat = int(match.group())
                del lines[i]
                break

        self.ntypat = ntypat
        if self.ntypat is None:
            raise Exception("ntypat is missing in the Abinit file!")

        # Extract znucl
        znucl = []
        for i, line in enumerate(lines):
            if line.strip().startswith('znucl'):
                znucl = list(map(int, re.findall(r"\d+", line)))  # Fixed typo in re.finall to re.findall
                del lines[i]
                break
        
        self.znucl = znucl
        if not znucl:  # Check if znucl is still an empty list
            raise Exception("znucl is missing in the Abinit file!")

        # Extract typat
        typat = []
        for i, line in enumerate(lines):
            if line.strip().startswith('typat'):
                # TODO: Length may be hardcoded
                typat_tokens = re.findall(r"(\d+\*\d+|\d+)", line)
                for token in typat_tokens:
                    if '*' in token:
                        count, value = map(int, token.split('*'))
                        typat.extend([value] * count)
                    else:
                        typat.append(int(token))
                del lines[i]
                break
        
        self.typat = typat
        if not typat:  # Check if typat is still an empty list
            raise Exception("typat is missing in the Abinit file!")

        # Save the remaining content as the header
        header_path = self.abi_file + ".header"
        self.header_path = header_path

        with open(header_path, "w") as header_file:
            header_file.writelines(lines)
        
        header_file.close()


    def _close_to_zero(self, value, threshold=1e-10):
        """Helper function to treat small numbers as zero."""
        return np.abs(value) < threshold

    def convertToXred(self):
        """Converts coordinates from cartesian to reduced"""
        if self.coord_type == 'reduced':
            print("The unit cell is already expressed in reduced coordinates")
        else:
            if not isinstance(self.coordinates, np.ndarray):
                self.coordinates = np.array(self.coordinates)
            assert len(self.coordinates.shape) == 2 and self.coordinates.shape[0] == self.num_atoms and self.coordinates.shape[1] == 3

            self.coord_type = 'reduced'
            # Calculate the reduced coordinates of the system
            rprim_scaled = self.rprim.copy()
            # Scaling the primitive vectors by the lattice constants
            for i in range(3):
                rprim_scaled[i] *= self.acell[i]
            
            # Convert Cartesian to reduced coordinates
            xred = np.dot(self.coordinates, np.linalg.inv(rprim_scaled))
            
            # Set small values to zero
            xred[np.abs(xred) < 1e-10] = 0
            
            self.coordinates = np.array(xred)
            print("Converted to reduced coordinates.")

    def convertToXcart(self):
        """Converts coordinates from reduced to cartesian"""
        if self.coord_type == 'cartesian':
            print("The unit cell is already expressed in cartesian coordinates")
        else:
            if not isinstance(self.coordinates, np.ndarray):
                self.coordinates = np.array(self.coordinates)
            assert len(self.coordinates.shape) == 2 and self.coordinates.shape[0] == self.num_atoms and self.coordinates.shape[1] == 3

            self.coord_type = 'cartesian'
            # Calculate the cartesian coordinates of the system
            rprim_scaled = self.rprim.copy()
            # Scaling the primitive vectors by the lattice constants
            for i in range(3):
                rprim_scaled[i] *= self.acell[i]
            
            # Convert reduced to Cartesian coordinates
            xcart = np.dot(self.coordinates, rprim_scaled)
            
            # Set small values to zero
            xcart[np.abs(xcart) < 1e-10] = 0
            
            self.coordinates = np.array(xcart)
            print("Converted to Cartesian coordinates.")
    
    def findSpaceGroup(self):
        """
        Calculates and returns the space group of the unit cell. Ensure coordinates are in cartesian

        Returns: 
            str: The space group symbol (e.g. 'Pm-3m') and international number.
        """

        # Scale the lattice vectors by acell 
        lattice = Lattice(self.rprim * self.acell[:, None])

        # Map typat and znucl to get the atomic numbers
        species = [self.znucl[typ - 1] for typ in self.typat]

        # Create the pymatgen Structure
        structure = Structure(
            lattice=lattice,
            species=species, 
            coords=self.coordinates, 
            coords_are_cartesian=(self.coord_type=="cartesian")

        )

        analyzer = SpacegroupAnalyzer(structure)
        space_group = analyzer.get_space_group_symbol()
        space_group_number = analyzer.get_space_group_number()

        return f"{space_group} ({space_group_number})"

    def perturbations(self, pert):
        """
        Apply a given perturbation to the unit cell coordinates and return a new UnitCell.
        
        Args:
            pert (np.ndarray): A numpy array representing the perturbation to be applied.

        Returns:
            UnitCell: A new instance of UnitCell with perturbed coordinates.
        """

        # Ensure the perturbation has the correct shape
        perturbation = np.array(pert, dtype=float)
        if perturbation.shape != self.coordinates.shape:
            raise ValueError("Perturbation must have the same shape as the coordinates.")

        # Calculate new coordinates by adding the perturbation
        new_coordinates = self.coordinates + perturbation

        # Create a new UnitCell object with the updated coordinates
        # Assuming other properties remain the same; adjust as needed
        perturbed_unit_cell = UnitCell(
            acell=self.acell,
            rprim=self.rprim,
            coordinates=new_coordinates,
            coord_type=self.coord_type,
            num_atoms=self.num_atoms,
            ntypat=self.ntypat,
            znucl=self.znucl,
            typat=self.typat,
        )

        return perturbed_unit_cell

    def __repr__(self):
        return (f"UnitCell(acell={self.acell}, coord_type='{self.coord_type}', rprim={self.rprim}, coordinates={self.coordinates}, "
                f"num_atoms={self.num_atoms}, znucl={self.znucl}, typat={self.typat}, "
                f"header_path='{self.header_path}')")


