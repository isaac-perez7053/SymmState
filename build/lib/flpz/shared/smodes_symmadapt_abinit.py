import numpy as np
import subprocess
import sys
import os
from pathlib import Path

# Constants
np.set_printoptions(precision=10)
dispMag = 0.001  # In angstroms
symmPrec = 1e-5  # Symmetry precision
atomList = [
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

def process_smodes(smodes_file, target_irrep, smodes_path="/Users/isaacperez/Downloads/Personal_Projects/isobyu/smodes"):
    """
    Process SMODES file and target irrep to generate symmetry-adapted modes.

    Args:
        smodes_file (str or Path): Path to the SMODES input file.
        target_irrep (str): Target irrep name.
        smodes_path (str or Path): Path to the SMODES binary. Defaults to "/home/iperez/isobyu/smodes".

    Returns:
        Path: Path to the generated header file.
    """
    smodes_file = Path(smodes_file).resolve()
    smodes_path = Path(smodes_path).resolve()

    # Validate paths
    if not smodes_file.exists():
        raise FileNotFoundError(f"SMODES input file not found: {smodes_file}")

    if not smodes_path.exists() or not smodes_path.is_file() or not os.access(smodes_path, os.X_OK):
        raise FileNotFoundError(f"SMODES binary not found or not executable: {smodes_path}")

    prec_lat_param, smodes_lines = read_smodes_file(smodes_file)
    print(f"PrecLatParam: {prec_lat_param}")

    smodes_output = run_smodes(smodes_path, smodes_file)
    target_output = extract_target_irrep(smodes_output, target_irrep)
    target_output = clean_target_output(target_output)

    shape_cell, atom_list, atom_pos_raw = parse_cell_data(target_output, prec_lat_param)

    # Normalize and clean the cell
    clean_list = [1 / 3, 2 / 3, np.sqrt(3), 1 / 6, 1 / 8, np.sqrt(2)]
    shape_cell = normalize_and_clean_matrix(shape_cell, clean_list, symmPrec)
    cell = np.multiply(shape_cell, np.array(prec_lat_param))
    print(f"Cell:\n{cell}")

    # Calculate atom types and counts
    type_list = list(set(atom_list))
    type_count = [atom_list.count(t) for t in type_list]

    # Get atomic masses
    atomic_masses = ["Unknown" for _ in type_list]

    # Write header file
    output_dir = smodes_file.parent / f"SMODES_{target_irrep}"
    if output_dir.exists():
        print(f"Warning: Output directory {output_dir} already exists. Overwriting.")
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                os.rmdir(item)
    output_dir.mkdir(exist_ok=True)

    header_name = output_dir / f"headerFile_{target_irrep}.dat"
    write_header_file(header_name, target_irrep, len(atom_list), type_list, type_count, atomic_masses)
    return header_name

def read_smodes_file(smodes_file):
    """Read the SMODES input file."""
    with open(smodes_file, 'r') as f:
        lines = f.readlines()
    prec_lat_param = list(map(float, lines[0].split()[:3]))
    return prec_lat_param, lines

def run_smodes(smodes_path, smodes_file):
    """Run the SMODES binary and return its parsed output."""
    args = f'{smodes_path} < {smodes_file}'
    proc = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
    output = proc.stdout.read().decode('ascii')
    return output.split("\n")

def extract_target_irrep(output_lines, target_irrep):
    """Extract the target irrep from SMODES output."""
    start_target, end_target = 999, 0
    for idx, line in enumerate(output_lines):
        words = line.split()
        if len(words) > 1 and words[0] == "Irrep" and words[1] == target_irrep:
            start_target = idx
        if len(words) > 0 and start_target < 999:
            if words[0] == "***********************************************":
                end_target = idx
                break
    return output_lines[start_target:end_target]

def clean_target_output(target_output):
    """Clean the target output by removing unnecessary lines."""
    if target_output[3].split()[0] in ['These', 'IR', 'Raman']:
        del target_output[3]
    return target_output

def parse_cell_data(target_output, prec_lat_param):
    """Parse cell data and atom positions."""
    v1, v2, v3 = (list(map(float, target_output[i].split())) for i in range(4, 7))
    shape_cell = np.matrix([v1, v2, v3])
    
    atom_list, atom_pos_raw = [], []
    for line in target_output[8:]:
        words = line.split()
        if len(words) < 4:
            break
        atom_list.append(words[1])
        atom_pos_raw.append([float(words[2]), float(words[3]), float(words[4])])

    return shape_cell, atom_list, atom_pos_raw

def normalize_and_clean_matrix(matrix, clean_list, precision):
    """Normalize and clean irrational numbers in a matrix."""
    for n in range(matrix.shape[0]):
        for i in range(matrix.shape[1]):
            for c in clean_list:
                if abs(abs(matrix[n, i]) - abs(c)) < precision:
                    matrix[n, i] = np.sign(matrix[n, i]) * c
    return matrix

def write_header_file(header_name, target_irrep, num_modes, atom_types, atom_counts, atom_masses):
    """Write header file with details about the symmetry-adapted modes."""
    with open(header_name, 'w') as h:
        h.write(f"Irrep: {target_irrep}\n")
        h.write(f"NumSAM: {num_modes}\n")
        h.write(f"NumAtomTypes: {len(atom_types)}\n")
        h.write(f"DispMag(angstrom): {dispMag}\n")
        for atom, count, mass in zip(atom_types, atom_counts, atom_masses):
            h.write(f"{atom} {int(count)} {mass}\n")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python smodes_symmadapt_abinit.py <smodes_file> <target_irrep>")
        sys.exit(1)

    smodes_file = Path(sys.argv[1])
    target_irrep = sys.argv[2]

    try:
        header_file = process_smodes(smodes_file, target_irrep)
        print(f"Header file created: {header_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
