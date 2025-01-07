import numpy as np
import os
from pathlib import Path

np.set_printoptions(precision=10)

def read_header_file(target_irrep):
    """Read the header file and extract key data."""
    header_file = Path.cwd() / f"SMODES_{target_irrep}" / f"headerFile_{target_irrep}.dat"
    
    if not header_file.exists():
        raise FileNotFoundError(f"Header file not found: {header_file}")

    with open(header_file, 'r') as f:
        header_lines = f.readlines()

    irrep = header_lines[0].split()[1]
    num_sam = int(header_lines[1].split()[1])
    num_atom_types = int(header_lines[2].split()[1])
    num_atoms = int(header_lines[3].split()[1])
    disp_mag = float(header_lines[4].split()[1])

    type_list, type_count, mass_list = [], [], []
    for line in header_lines[5:5 + num_atom_types]:
        words = line.split()
        type_list.append(words[0])
        type_count.append(int(words[1]))
        mass_list.append(float(words[2]) if words[2] != "MASS" else None)

    if None in mass_list:
        raise ValueError(f"Mass not set in {header_file}! Cannot proceed.")

    return irrep, num_sam, num_atom_types, num_atoms, disp_mag, type_list, type_count, mass_list, header_lines

def parse_sam_data(header_lines, num_atom_types, num_atoms, num_sam):
    """Parse symmetry-adapted mode (SAM) data from header lines."""
    sam_mat = np.zeros((num_atoms, 3, num_sam + 1))
    sam_atom_label = []

    mode_index, atom_ind = 0, 0
    for line in header_lines[5 + num_atom_types:]:
        words = line.split()
        if words[0].startswith("SAM"):
            mode_index += 1
            atom_ind = 0
            sam_atom_label.append(words[1])
        else:
            sam_mat[atom_ind, 0, mode_index] = float(words[0])
            sam_mat[atom_ind, 1, mode_index] = float(words[1])
            sam_mat[atom_ind, 2, mode_index] = float(words[2])
            atom_ind += 1

    return sam_mat[:, :, 1:], sam_atom_label

def read_forces(num_atoms, num_sam, target_irrep):
    """Read force data from the output files."""
    force_mat_raw = np.zeros((num_atoms, 3, num_sam + 1))
    for sam in range(num_sam + 1):
        outcar_file = Path.cwd() / f"SMODES_{target_irrep}" / f"dist_{sam}" / f"dist_{sam}.abo"
        
        if not outcar_file.exists():
            raise FileNotFoundError(f"OUTCAR file not found: {outcar_file}")

        with open(outcar_file, 'r') as f:
            lines = f.readlines()

        # Find start of forces
        line_start = next(
            i + 1 for i, line in enumerate(lines)
            if "cartesian forces" in line
        )

        for i, line in enumerate(lines[line_start:line_start + num_atoms]):
            words = line.split()
            force_mat_raw[i, :, sam] = list(map(float, words[1:4]))

    return force_mat_raw

def compute_force_matrix(force_mat_raw, sam_mat, num_atoms, num_sam):
    """Compute the force matrix."""
    force_list = force_mat_raw[:, :, 1:] - force_mat_raw[:, :, [0]]
    force_mat = np.zeros((num_sam, num_sam))
    
    for f in range(num_sam):
        for s in range(num_sam):
            force_val = np.multiply(force_list[:, :, f], sam_mat[:, :, s])
            force_mat[f, s] = force_val.sum()

    return force_mat

def compute_mass_matrix(type_list, mass_list, sam_atom_label, num_sam):
    """Compute the mass matrix."""
    mass_vec = np.array([
        mass_list[type_list.index(label)] for label in sam_atom_label
    ])

    mm = np.sqrt(np.outer(mass_vec, mass_vec))
    return mm

def calculate_dynamical_matrix(force_mat, mass_matrix):
    """Calculate the dynamical matrix and its eigenvalues/eigenvectors."""
    dyn_mat = force_mat / mass_matrix
    evals, evecs = np.linalg.eig(dyn_mat)
    return evals, evecs, dyn_mat

def write_results(target_irrep, freq_thz, freq_cm, dynevecs):
    """Write results to files in the current working directory."""
    output_dir = Path.cwd() / f"SMODES_{target_irrep}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "DynFreqs.dat", 'w') as f:
        f.write("THz \t cm^-1 \n")
        for thz, cm in zip(freq_thz, freq_cm):
            f.write(f"{thz:.2f}\t{cm:.2f}\n")

    with open(output_dir / "DynEvecs.dat", 'w') as f:
        for evec in dynevecs:
            f.write("\t".join(map("{:.5f}".format, evec.flatten())) + "\n")

def process_smodesPost(target_irrep):
    """Process SMODES data and generate outputs."""
    irrep, num_sam, num_atom_types, num_atoms, disp_mag, type_list, type_count, mass_list, header_lines = read_header_file(target_irrep)
    sam_mat, sam_atom_label = parse_sam_data(header_lines, num_atom_types, num_atoms, num_sam)
    force_mat_raw = read_forces(num_atoms, num_sam, target_irrep)

    force_mat = compute_force_matrix(force_mat_raw, sam_mat, num_atoms, num_sam)
    mass_matrix = compute_mass_matrix(type_list, mass_list, sam_atom_label, num_sam)
    evals, evecs, dyn_mat = calculate_dynamical_matrix(force_mat, mass_matrix)

    freq_thz = np.sign(evals) * np.sqrt(np.abs(evals)) * 1e-12
    freq_cm = freq_thz * 1e12 / (2.9979e10)

    print("Mass Matrix:")
    print(mass_matrix)
    print("\nForce Matrix:")
    print(force_mat)
    print("\nDynamical Matrix:")
    print(dyn_mat)

    write_results(target_irrep, freq_thz, freq_cm, evecs)

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python smodes_postproc_abinit.py <target_irrep>")
        sys.exit(1)

    target_irrep = sys.argv[1]
    process_smodesPost(target_irrep)

if __name__ == "__main__":
    main()

