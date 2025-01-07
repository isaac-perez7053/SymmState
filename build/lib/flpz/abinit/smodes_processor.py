import numpy as np
import os
import subprocess
import sys
from pathlib import Path

from . import AbinitUnitCell
np.set_printoptions(precision=10)

import tracemalloc
tracemalloc.start()

class SmodesProcessor(AbinitUnitCell):
    """
    Uses SMODES to c
    """
    def __init__(self, abi_file, smodes_input, target_irrep, symm_prec=1e-5, disp_mag=0.001, smodes_path='../isobyu/smodes', host_spec='mpirun -hosts=localhost -np 30', b_script_header_file=None):
        super().__init__(abi_file=abi_file, batch_script_header_file=b_script_header_file)
        self.target_irrep = target_irrep
        self.smodes_path = smodes_path
        self.symm_prec = symm_prec
        self.disp_mag = disp_mag
        self.host_spec = host_spec

        # Initialize attributes to be populated by _create_header
        self.irrep = None
        self.num_sam = None
        self.sam_atom_label =[]
        self.mass_list = []
        self.dist_mat = None
        self.pos_mat_cart = None
        self.type_count = None

        self.isir = False
        self.israman = False
        self.transmodes = False
        self.crossdot_ispos = False

        self.mass_matrix = None
        self.springs_constants_matrix = None

        # Populate header information
        self._create_header(smodes_input)
        
        # Attributes for storing calculated data
        self.jobs_ran_abo = []
        self.dyn_freqs = []
        self.fc_evals = None
        self.phonon_vecs = None
        self.red_mass = None

    def convertToXcart(self):
        return super().convertToXcart()
    
    def convertToXred(self):
        return super().convertToXred()
    
    def findSpaceGroup(self):
        return super().findSpaceGroup()
    
    def write_custom_abifile(self, output_file, header_file):
        return super().write_custom_abifile(output_file, header_file)

    def _create_header(self, smodes_input):
        """Extract header information and store in attributes."""
        atom_list = [
            ('1', 'H', '1.008'), ('2', 'He', '4.002'), ('3', 'Li', '6.94'), ('4', 'Be', '9.012'),
            ('5', 'B', '10.81'), ('6', 'C', '12.011'), ('7', 'N', '14.007'), ('8', 'O', '15.999'),
            ('9', 'F', '18.998'), ('10', 'Ne', '20.180'), ('11', 'Na', '22.990'), ('12', 'Mg', '24.305'),
            ('13', 'Al', '26.982'), ('14', 'Si', '28.085'), ('15', 'P', '30.974'), ('16', 'S', '32.06'),
            ('17', 'Cl', '35.45'), ('18', 'Ar', '39.948'), ('19', 'K', '39.098'), ('20', 'Ca', '40.078'),
            ('21', 'Sc', '44.956'), ('22', 'Ti', '47.867'), ('23', 'V', '50.942'), ('24', 'Cr', '51.996'),
            ('25', 'Mn', '54.938'), ('26', 'Fe', '55.845'), ('27', 'Co', '58.933'), ('28', 'Ni', '58.693'),
            ('29', 'Cu', '63.546'), ('30', 'Zn', '65.38'), ('31', 'Ga', '69.723'), ('32', 'Ge', '72.630'),
            ('33', 'As', '74.922'), ('34', 'Se', '78.971'), ('35', 'Br', '79.904'), ('36', 'Kr', '83.798'),
            ('37', 'Rb', '85.468'), ('38', 'Sr', '87.62'), ('39', 'Y', '88.906'), ('40', 'Zr', '91.224'),
            ('41', 'Nb', '92.906'), ('42', 'Mo', '95.95'), ('43', 'Tc', '98'), ('44', 'Ru', '101.07'),
            ('45', 'Rh', '102.91'), ('46', 'Pd', '106.42'), ('47', 'Ag', '107.87'), ('48', 'Cd', '112.41'),
            ('49', 'In', '114.82'), ('50', 'Sn', '118.71'), ('51', 'Sb', '121.76'), ('52', 'Te', '127.60'),
            ('53', 'I', '126.90'), ('54', 'Xe', '131.29'), ('55', 'Cs', '132.91'), ('56', 'Ba', '137.33'),
            ('57', 'La', '138.91'), ('58', 'Ce', '140.12'), ('59', 'Pr', '140.91'), ('60', 'Nd', '144.24'),
            ('61', 'Pm', '145'), ('62', 'Sm', '150.36'), ('63', 'Eu', '151.96'), ('64', 'Gd', '157.25'),
            ('65', 'Tb', '158.93'), ('66', 'Dy', '162.50'), ('67', 'Ho', '164.93'), ('68', 'Er', '167.26'),
            ('69', 'Tm', '168.93'), ('70', 'Yb', '173.05'), ('71', 'Lu', '174.97'), ('72', 'Hf', '178.49'),
            ('73', 'Ta', '180.95'), ('74', 'W', '183.84'), ('75', 'Re', '186.21'), ('76', 'Os', '190.23'),
            ('77', 'Ir', '192.22'), ('78', 'Pt', '195.08'), ('79', 'Au', '196.97'), ('80', 'Hg', '200.59'),
            ('81', 'Tl', '204.38'), ('82', 'Pb', '207.2'), ('83', 'Bi', '208.98'), ('84', 'Po', '209'),
            ('85', 'At', '210'), ('86', 'Rn', '222'), ('87', 'Fr', '223'), ('88', 'Ra', '226'),
            ('89', 'Ac', '227'), ('90', 'Th', '232.04'), ('91', 'Pa', '231.04'), ('92', 'U', '238.03'),
            ('93', 'Np', '237'), ('94', 'Pu', '244'), ('95', 'Am', '243'), ('96', 'Cm', '247'),
            ('97', 'Bk', '247'), ('98', 'Cf', '251'), ('99', 'Es', '252'), ('100', 'Fm', '257'),
            ('101', 'Md', '258'), ('102', 'No', '259'), ('103', 'Lr', '266'), ('104', 'Rf', '267'),
            ('105', 'Db', '268'), ('106', 'Sg', '269'), ('107', 'Bh', '270'), ('108', 'Hs', '277'),
            ('109', 'Mt', '278'), ('110', 'Ds', '281'), ('111', 'Rg', '282'), ('112', 'Cn', '285'),
            ('113', 'Nh', '286'), ('114', 'Fl', '289'), ('115', 'Mc', '290'), ('116', 'Lv', '293'),
            ('117', 'Ts', '294'), ('118', 'Og', '294')
        ]


        if not Path(self.smodes_path).is_file():
            raise FileNotFoundError(f"SMODES executable not found at: {self.smodes_path}. Current directory is {os.getcwd()}")

        # Open and read SMODES input file
        with open(smodes_input) as s:
            s_lines = s.readlines()
    

        # Parse lattice parameters
        prec_lat_param = [float(x) for x in s_lines[0].split()]
        # I need to check if this is correct
        self.acell = [1, 1, 1]

        # Execute SMODES and process output
        command = f"{self.smodes_path} < {smodes_input}"
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

        # The computer doesn't like if I don't wait :3 
        proc.wait()
        output = proc.stdout.read().decode('ascii')
        proc.stdout.close()

        # Process the output from SMODES
        start_target = 999
        end_target = 0
        outlist = output.split("\n")

        for line in range(len(outlist)):
            line_content = outlist[line].split()
            if len(line_content) > 1 and line_content[0] == "Irrep" and line_content[1] == self.target_irrep:
                start_target = line

            if len(line_content) > 0 and start_target < 999:
                if line_content[0] == "***********************************************":
                    end_target = line
                    break

        target_output = outlist[start_target:end_target]

        if (target_output[3].split()[0]=='These'):
            self.transmodes = True
            del target_output[3] 

        if (target_output[3].split()[0]=='IR'):
            self.isir = True
            del target_output[3] 

        if (target_output[3].split()[0]=='Raman'):
            self.israman = True
            del target_output[3] 


        # Parse degeneracy and number of modes
        degeneracy = int(target_output[1].split()[-1])
        num_modes_without_degen = int(target_output[2].split()[-1])
        num_modes = num_modes_without_degen // degeneracy

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
                atom_positions_raw.append([float(line_content[2]), float(line_content[3]), float(line_content[4])])

        # Number of atoms (Begin updating the Abinit file with it's new attributes )
        self.natom = len(atom_names)

        # Find number of atom types and count them
        self.ntypat = len(list(set(atom_names)))

        # Dictionary to count the occurances of each element
        count_dict = {}
        result = []

        # Iterate over each element in the input array
        for item in list(atom_names):
            # Increment the count of the element
            if item in count_dict:
                count_dict[item] += 1
            else:
                count_dict[item] = 1
        
        # Create a list of multiplicities based on the original order
        multiplicity_list = []
        # Set to keep track of already processed elements
        seen = set()

        for item in list(atom_names):
            if item not in seen:
                multiplicity_list.append(count_dict[item])
                seen.add(item)

        self.type_count = multiplicity_list

        for index, value in enumerate(multiplicity_list):
            result.extend([(index + 1)] * value)

        self.typat = result 


        # Clean the cell for possible irrational numbers and clean fractions
        clean_list = self._generate_clean_list()
        shape_cell = self._clean_matrix(shape_cell, clean_list)

        # Extract the cell
        prec_lat_array = np.array([prec_lat_param, prec_lat_param, prec_lat_param])
        self.rprim = np.multiply(shape_cell, prec_lat_array)

        # Clean up atom positions
        atom_positions = self._clean_positions(atom_positions_raw, prec_lat_param, clean_list)
        self.coordinates = np.array(atom_positions)
        self.coord_type = 'cartesian'

        # Convert to Cartesian coordinates
        pos_mat_cart = np.array(atom_positions)
        
        # Return list of atom names without duplicates
        seen = set()
        atom_names_nodup = []

        for item in atom_names:
            if item not in seen:
                atom_names_nodup.append(item)
                seen.add(item)
        self.type_list = atom_names_nodup

        # Find all the masses of the atoms
        atomic_num_list, atomic_mass_list = self._get_atomic_details(atom_names_nodup, atom_list)

        self.znucl = atomic_num_list

        # Store extracted data as class attributes
        self.irrep = self.target_irrep
        self.num_sam = num_modes
        self.mass_list = atomic_mass_list
        self.pos_mat_cart = pos_mat_cart

        # Calculate the displacement matrix
        start_line = self.natom + 11
        dist_mat = self._calculate_displacement_matrix(target_output, num_modes, self.natom, start_line)
        # Normalize and orthogonalize SAMs
        self.dist_mat = self._orthogonalize_sams(dist_mat, num_modes, self.natom)

        # Abinit requires this be positive
        crossDot = np.dot(np.cross(self.rprim[0,:], self.rprim[1,:]),np.transpose(self.rprim[2,:]))
        self.crossdot_ispos = crossDot > 0



    def _generate_clean_list(self):
        """Helper function to generate a list of rational approximations."""
        clean_list = [1.0 / 3.0, 2.0 / 3.0]
        for i in range(1, 10):
            for base in [np.sqrt(3), np.sqrt(2)]:
                clean_list.extend([
                    base / float(i), 2 * base / float(i), 3 * base / float(i),
                    4 * base / float(i), 5 * base / float(i), float(i) / 6.0, float(i) / 8.0
                ])
        return clean_list

    def _clean_matrix(self, matrix, clean_list):
        """Clean a matrix by replacing approximate values with exact ones using clean_list."""
        for n in range(matrix.shape[0]):
            for i in range(matrix.shape[1]):
                for c in clean_list:
                    if abs(abs(matrix[n, i]) - abs(c)) < self.symm_prec:
                        matrix[n, i] = np.sign(matrix[n, i]) * c
        return matrix

    def _clean_positions(self, positions, prec_lat_param, clean_list):
        """Clean atomic positions and convert using lattice parameters."""
        # Copy positions to avoid modifying the original input data
        cleaned_positions = positions.copy()

        for n, pos in enumerate(cleaned_positions):
            for i in range(3):
                for c in clean_list:
                    if abs(abs(pos[i]) - abs(c)) < self.symm_prec:
                        pos[i] = np.sign(pos[i]) * c
                pos[i] *= prec_lat_param[i]

        # Convert to a NumPy array to ensure consistent processing
        cleaned_positions = np.array(cleaned_positions)

        # Ensure dimensions are correct 
        if cleaned_positions.ndim != 2 or cleaned_positions.shape[1] != 3:
            raise ValueError(f"Cleaned positions do not have expected shape (n_atoms, 3): {cleaned_positions.shape}")

        return np.array(cleaned_positions)


    def _get_atomic_details(self, type_list, atom_list):
        """Retrieve atomic numbers and masses based on atom type."""
        atom_dict = {record[1]: (record[0], float(record[2])) for record in atom_list}

        atomic_num_list = []
        atomic_mass_list = []

        for atom_type in type_list:
            if atom_type in atom_dict:
                atomic_number, atomic_mass = atom_dict[atom_type]
                atomic_num_list.append(atomic_number)
                atomic_mass_list.append(atomic_mass)
            else:
                print(f"Unrecognized atom {atom_type}, manual entry required")
                atomic_num_list.append(0)
                atomic_mass_list.append(0)

        return atomic_num_list, atomic_mass_list

        
    def _calculate_displacement_matrix(self, target_output, num_modes, num_atoms, start_line):
        """Calculate the initial displacement matrix from SMODES output."""
        dist_mat = np.zeros((num_modes, num_atoms, 3))
        mode_index = -1
        self.sam_atom_label = [None] * num_modes

        for l in range(start_line, len(target_output)):
            line_content = target_output[l].split()
            if target_output[l] == '------------------------------------------':
                mode_index += 1
                
            else:
                atom = int(line_content[0]) - 1
                self.sam_atom_label[mode_index] = line_content[1]
                disp1, disp2, disp3 = map(float, line_content[2:5])
                dist_mat[mode_index, atom, 0] = disp1
                dist_mat[mode_index, atom, 1] = disp2
                dist_mat[mode_index, atom, 2] = disp3
        
        return dist_mat

    def _orthogonalize_sams(self, dist_mat, num_modes, num_atoms):
        """Normalize and orthogonalize the systematic atomic modes (SAMs)."""
        # Normalize the SAMs
        for m in range(0, num_modes):
            norm = np.linalg.norm(dist_mat[m, :, :])
            if norm == 0:
                raise ValueError(f"Zero norm encountered at index {m} during normalization.")
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
                print(f"Warning: Zero norm encountered at index {m} during orthogonalization.")

        return orth_mat

    def _loop_modes(self):
        """
        # Creates the displaced cells and runs them through Abinit for the _perform_calculations method
        """
        content = """
getwfk 1
useylm 1
kptopt 2
chkprim 0
"""
        # Run the original unchanged cell
        self.convertToXred()
        original_coords = self.coordinates.copy()
        abi_name = "dist_0"
        self.write_custom_abifile(abi_name, content)
        self.run_abinit(input_file=abi_name, batch_name='dist_0_sbatch', batch_script_header_file=self.batchScriptHeader_path, host_spec=self.host_spec, log='dist_0.log')
        self.jobs_ran_abo.append(f"dist_0.abo")

        # Displace each cell
        for i in range(self.num_sam):
            j = i + 1

            # Calculate displacement
            self.coordinates = np.array(self.coordinates + (1.88973 * self.disp_mag * self.dist_mat[i]))
            abi_name = f"dist_{j}"
            batch_name = f"dist_{j}_sbatch"

            # Write abifile and run abinit
            self.write_custom_abifile(abi_name, content)
            self.run_abinit(input_file=abi_name, batch_name=batch_name, batch_script_header_file=self.batchScriptHeader_path, host_spec=self.host_spec, log=f"dist_{j}.log")

            # Change coordinates back to their original value
            self.coordinates = np.array(original_coords).copy()
            self.jobs_ran_abo.append(f"dist_{j}.abo")

        # Originally set to 300
        self.wait_for_jobs_to_finish(60)


    def _perform_calculations(self):
        """
        Calculates the eigen-frequencies associated with a particular representation
        """

        if not self.crossdot_ispos:
            raise ValueError("The cross (R1, R2)*R3 must be positive or Abinit will freak out")

        # The extra dimension is saved for the force matrix of the original cell
        force_mat_raw = np.zeros((self.num_sam + 1, self.natom, 3))
        
        for sam, abo in enumerate(self.jobs_ran_abo):
            with open(abo) as f:
                abo_lines = f.readlines()

            line_start = 0
            atom_ind = 0
            
            for line_num, line in enumerate(abo_lines):
                words = line.split()
                if len(words) >= 1 and words[0] == "cartesian" and words[1] == "forces" and words[2] == "(eV/Angstrom)":
                    line_start = line_num + 1
                    break
                    
            for line_num in range(line_start, line_start + self.natom):
                words = abo_lines[line_num].split()
                force_mat_raw[sam, atom_ind, 0] = float(words[1])
                force_mat_raw[sam, atom_ind, 1] = float(words[2])
                force_mat_raw[sam, atom_ind, 2] = float(words[3])
                atom_ind += 1

        print(f"Printing forceforce_mat_raw:\n \n {force_mat_raw} \n")

        # This will not include the original cell forces
        force_list = np.zeros((self.num_sam, self.natom, 3))
        
        # Subtract off the forces from the original cell (should hopefully be zero)
        for sam in range(self.num_sam):
            for i in range(self.natom):
                for j in range(3):
                    force_list[sam, i, j] = force_mat_raw[sam + 1, i, j] - force_mat_raw[0, i, j] 

        for i, matrix in enumerate(force_list):
            force_list[i] = np.roll(matrix, shift=-1, axis=1)
        print(f"Printing force list: \n \n {force_list} \n")
        force_matrix = np.zeros((self.num_sam, self.num_sam))

        # Calculate force matrix
        for f in range(self.num_sam):
            for s in range(self.num_sam):
                force_val = np.multiply(self.dist_mat[s, :, :], force_list[f, :, :])
                force_matrix[f, s] = force_val.sum()

        print(f"Printing force matrix: \n \n {force_matrix} \n")
        self.springs_constants_matrix = np.array(force_matrix)
        mass_vector = np.zeros((self.num_sam))
        
        # Calculate mass matrix
        for m in range(self.num_sam):
            this_mass = 0
            for n in range(self.ntypat):
                if self.sam_atom_label[m] == self.type_list[n]:
                    this_mass = self.mass_list[n]
                    mass_vector[m] = this_mass
            if this_mass == 0:
                raise ValueError("Problem with building mass matrix. Quitting...")

        mass_matrix = np.zeros((self.num_sam, self.num_sam))
        self.mass_matrix = np.array(mass_matrix)
        
        for m in range(self.num_sam):
            for n in range(self.num_sam):
                mass_matrix[m, n] = np.sqrt(mass_vector[m]) * np.sqrt(mass_vector[n])

        print(f"Printing mass matrix: \n \n {mass_matrix} \n")
        
        fc_mat = -force_matrix / self.disp_mag
        fc_mat = (fc_mat + np.transpose(fc_mat)) / 2.0
        dyn_mat = np.divide(fc_mat, mass_matrix)

        fc_evals, _ = np.linalg.eig(fc_mat)
        dynevals, dynevecs_sam = np.linalg.eig(dyn_mat)

        eV_to_J=1.602177E-19
        ang_to_m=1.0E-10
        AMU_to_kg=1.66053E-27
        c= 2.9979458E10 

        freq_thz = np.multiply(np.sign(dynevals), np.sqrt(np.absolute(dynevals) * eV_to_J / (ang_to_m ** 2 * AMU_to_kg)) * 1.0E-12)
        fc_eval = np.multiply(np.sign(fc_evals), np.sqrt(np.absolute(fc_evals)))

        idx_dyn = np.flip(freq_thz.argsort()[::-1])
        freq_thz = freq_thz[idx_dyn] / (2 * np.pi)
        dynevecs_sam = dynevecs_sam[:, idx_dyn]

        freq_cm = freq_thz * 1.0E12 / c

        self.dyn_freqs = [[freq_thz[i], freq_cm[i]] for i in range(self.num_sam)]
        self.fc_evals = fc_eval[idx_dyn]

        dynevecs = np.zeros((self.num_sam, self.natom, 3))
        
        for evec in range(self.num_sam):
            real_dynevec = np.zeros((self.natom, 3))
            for s in range(self.num_sam):
    ###########################################################################
    # I need to figure out the dimensions of these matricies
    ########################################################
                real_dynevec += dynevecs_sam[evec, s] * self.dist_mat[s, :, :]
            dynevecs[evec, :, :] = real_dynevec

        mass_col = np.zeros((self.natom, 3))
        atomind = 0
        
        for atype in range(self.ntypat):
            for j in range(self.type_count[atype]):
                mass_col[atomind, :] = np.sqrt(self.mass_list[atype])
                atomind += 1

        phon_disp_eigs = np.zeros((self.num_sam, self.natom, 3))
        redmass_vec = np.zeros((self.natom, 1))

        for mode in range(self.num_sam):
            phon_disp_eigs[mode, :, :] = np.divide(dynevecs[mode, :, :], mass_col)
            mag_squared = np.sum(np.sum(np.multiply(phon_disp_eigs[mode, :, :], phon_disp_eigs[mode, :, :])))
            redmass_vec[mode] = 1.0 / mag_squared
            phon_disp_eigs[mode, :, :] /= np.sqrt(mag_squared)

        self.phonon_vecs = phon_disp_eigs
        self.red_mass = redmass_vec

        
    def _imaginary_frequencies(self):
        """Detects whether imaginary frequencies exist"""
        # List to store indicies with negative freq_thz values
        negative_indicies = []

        # Iterate over dyn_freqs to check the first element of each sublist
        for index, (freq_thz, _) in enumerate(self.dyn_freqs):
            if freq_thz < 0:
                negative_indicies.append(index)
        
        # Return the list of indicies or False if none are negative
        return negative_indicies if negative_indicies else False


    def run_smodes(self, smodes_input):
        """Run the SMODES executable."""
        if not Path(self.smodes_path).is_file():
            raise FileNotFoundError(f"SMODES executable not found at: {self.smodes_path}")

        # Redirect output to the designated output directory
        command = f"{self.smodes_path} < {smodes_input} > output.log"
        process = subprocess.run(command, shell=True, capture_output=True, text=True)

        if process.returncode != 0:
            raise RuntimeError(f"SMODES execution failed: {process.stderr}")

        return process.stdout

    # TODO: check the vectors are normalized correctly
    def unstable_phonons(self):
        """Outputs the unstable phonons and false if none are present"""
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
                normalized_matrix = normalized_flattened.reshape(self.phonon_vecs[i].shape)
                unstable_phonons_normalized.append(normalized_matrix)

            return unstable_phonons_normalized

    def symmadapt(self):
        """Runs the symmadapt sub-program"""
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
