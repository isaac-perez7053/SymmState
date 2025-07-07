from symmstate.abinit import AbinitFile
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from typing import Dict, Any
import re
from collections import defaultdict
import subprocess
from symmstate.config.symm_state_settings import settings
from symmstate.utils import get_unique_filename


class IsoFile:
    """
    Class containing and handling on input and output files of the Isotropy program
    """

    # Open to including other popular DFT programs
    def __init__(self, structure_file: AbinitFile):
        self.abi_file = structure_file

    def write_isotropy_input(
        self, out_path: str, irrep: str, direction: str, domain: int
    ):
        """
        Generate an ISOTROPY “.txt” that, when piped into `iso < file`, will
        print the subgroup Wyckoff coordinates and projected vectors.
        """
        a, b, c = self.abi_file.lattice.abc
        alpha, beta, gamma = self.abi_file.lattice.angles
        parent_sg = self.abi_file.space_group[0]

        out_path = get_unique_filename(out_path)

        lines = [
            f"VALUE PARENT {parent_sg}",
            f"VALUE IRREP {irrep}",
            f"VALUE DIR {direction}",
            f"VALUE DOMAIN {domain}",
            f"VALUE LATTICE PARAMETER {a:.6f} {b:.6f} {c:.6f} {alpha:.6f} {beta:.6f} {gamma:.6f}",
            "",
            "SHOW DIRECTION VECTOR",
            "SHOW SUBGROUP",
            "SHOW BASIS",
            "SHOW WYCKOFF",
            "SHOW MICROSCOPIC VECTOR",
            "DISPLAY ISOTROPY",
            "",  # ← this blank line tells Fortran “we’re done”
        ]

        with open(out_path, "w") as f:
            f.write("\n".join(lines))

        return out_path

    def _symm_adapt_basis_input(
        self, output_file: str, desired_irrep: str, direction: str
    ):
        """
        Write the input file for the ISOTROPY program to extract the symmetry adapted basis
        """

        output_file = get_unique_filename(output_file)

        sga = SpacegroupAnalyzer(self.abi_file)
        symm_struct = sga.get_symmetrized_structure()

        wyckoff_lines = []
        for wyckoff_symbol, sites in zip(
            symm_struct.wyckoff_symbols, symm_struct.equivalent_sites
        ):
            for site in sites:
                wyckoff_lines.append(f"VALUE WYCKOFF {wyckoff_symbol.upper()}")
                wyckoff_lines.append(
                    f"VALUE WYCKOFF XYZ {' '.join(str(x) for x in site.frac_coords)}"
                )
                wyckoff_lines.append(f"DISPLAY DISTORTION")

        wyckoff_block = "\n".join(wyckoff_lines)

        content = f"""
VALUE PARENT {self.abi_file.space_group[0]}
VALUE IRREP {desired_irrep}
VALUE DIR {direction}
VALUE DOMAIN 1
VALUE LATTICE PARAMETER 1 1 1 {' '.join(str(round(a)) for a in self.abi_file.lattice.angles)}
SHOW DIRECTION VECTOR
SHOW SUBGROUP
SHOW BASIS
SHOW WYCKOFF
SHOW MICROSCOPIC VECTOR
DISPLAY ISOTROPY

{wyckoff_block}

"""
        try:
            with open(output_file, "w") as f:
                f.write(content)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        print("Symm adapt basis input written successfuly!")

        return output_file

    def extract_wyckoff_points(self, iso_input_path: str, iso_exe: str = "iso"):
        """
        Run `iso < iso_input_path`, parse its “Dir Domain Wyckoff Point …” blocks,
        and return a dict mapping each Wyckoff letter ('a','i','d','c',…) to an
        (N×3) NumPy array of fractional coordinates.
        """
        # 1) call iso
        cmd = f"{iso_exe} < {iso_input_path}"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        out = result.stdout

        # 2) regexes
        #   - entry_line picks up lines that introduce a new Wyckoff letter
        entry_line = re.compile(r"^\s*\S+\s+\d+\s+([a-zA-Z])\s*\(")
        #   - coord finds every (x, y, z) in the text
        coord = re.compile(r"\(\s*([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\)")

        points = defaultdict(list)
        current_letter = None

        for line in out.splitlines():
            m = entry_line.match(line)
            if m:
                # start of a new letter block
                current_letter = m.group(1).lower()  # e.g. 'i', 'd', 'c'
            if current_letter:
                for x, y, z in coord.findall(line):
                    points[current_letter].append([float(x), float(y), float(z)])

        # 3) convert to NumPy arrays
        for letter, lst in points.items():
            points[letter] = np.array(lst)

        return points

    def symm_adapt_basis(
        self, desired_irrep: str, direction: str, disp_mag: float = 0.1
    ):
        """
        Calculate the symmetry adapted basis
        """

        np.set_printoptions(precision=10)
        print("The parent basis is: ")
        print()
        for row in self.abi_file.lattice.matrix:
            print(f"   {'  '.join(map(str, row))}\n")

        parent_basis = self.abi_file.lattice.matrix
        type_list = self.abi_file.typat_to_elem()
        natom = self.abi_file.vars["natom"]
        type_count = self.abi_file.element_multiplicity()

        # TODO: Ask Prof. Ritz what this is
        domain_basis = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        wyckXYZ_i = 0.2480103425295411
        wyckXYZ_e = 0.3207760730752693
        aPoints = np.matrix([[0.0000000000, 0.0000000000, 0.0000000000]])
        iPoints = np.matrix(
            [
                [0.5000000000, 0.0000000000, wyckXYZ_i],
                [0.5000000000, 0.5000000000, wyckXYZ_i],
                [0.0000000000, 0.5000000000, wyckXYZ_i],
                [0.5000000000, 0.0000000000, -wyckXYZ_i],
                [0.5000000000, 0.5000000000, -wyckXYZ_i],
                [0.0000000000, 0.5000000000, -wyckXYZ_i],
            ]
        )
        ePoints = np.matrix(
            [
                [0.0000000000, 0.0000000000, wyckXYZ_e],
                [0.0000000000, 0.0000000000, -wyckXYZ_e],
            ]
        )
        dPoints = np.matrix(
            [
                [0.3333333333, 0.6666666666, 0.5000000000],
                [-0.3333333333, 0.3333333333, 0.5000000000],
            ]
        )
        cPoints = np.matrix(
            [
                [0.3333333333, 0.6666666666, 0.0000000000],
                [-0.3333333333, 0.3333333333, 0.0000000000],
            ]
        )
        new_frac = np.matrix([[0, 0, 0]])
        modeTypeList = (
            ["dummy"] + 0 * 2 * ["Sc"] + 2 * 2 * ["V"] + 1 * 2 * ["Sn"]
        )  # has to be in order of iso input

        # # 1. write your iso input as before
        # self.write_isotropy_input(
        #     "iInput.txt", parent_sg, irrep, direction, domain, abc, angles
        # )

        # 2. extract the Wyckoff points
        wyck = self.extract_wyckoff_points("iInput.txt")

        # 3. plug them into your workflow:
        aPoints = wyck.get("a", np.empty((0, 3)))
        iPoints = wyck.get("i", np.empty((0, 3)))
        ePoints = wyck.get("e", np.empty((0, 3)))
        dPoints = wyck.get("d", np.empty((0, 3)))
        cPoints = wyck.get("c", np.empty((0, 3)))

        # Generate the isotropy input file
        iso_file = self._symm_adapt_basis_input(
            desired_irrep=desired_irrep, direction=direction
        )
        point_list = [aPoints, iPoints, ePoints, dPoints, cPoints]

        # Generate supercell
        super_cell = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                super_cell[i, :] = super_cell[i, :] + domain_basis[i, j] * parent_basis

        print("The supercell that was generated was: ")
        print()
        for row in super_cell:
            print(f"   {'  '.join(map(str, row))}\n")

        post_frac_parent = np.zeros((natom, 3))
        atom_ind = 0
        for type in range(len(point_list)):
            parent_site = point_list[type]
            shape_site = parent_site.shape
            for site in range(shape_site[0]):
                for shift in range(new_frac.shape[0]):
                    new_pos = parent_site[site, :] + new_frac[shift]
                    post_frac_parent[atom_ind, :] = new_pos
                    atom_ind += 1
                    print(f"Current atom indice: {atom_ind}")

        print("Post parent structure: ")
        print()
        for row in post_frac_parent:
            print(f"   {'  '.join(map(str, row))}\n")

        atom_cart = np.matmul(post_frac_parent, parent_basis)
        atom_frac = np.matmul(atom_cart, np.linalg.inv(super_cell))

        result = subprocess.run(
            ["iso <", iso_file], capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()
        out_list = output.split("\n")

        cleaned_lines0 = []
        for l in range(len(out_list)):
            this_line = out_list[l].split()
            if (this_line[0] == "Enter") and (this_line[1] == "RETURN"):
                pass
            else:
                cleaned_lines0.append(out_list[l])

        cleaned_lines1 = []
        last_line = ""
        for l in range(len(cleaned_lines0)):
            this_line = cleaned_lines0[l].split()
            if this_line[-1][-1] == ",":
                last_line = last_line + cleaned_lines0[l].lstrip(" ")
            else:
                cleaned_lines1.append(
                    (
                        "".join(last_line + cleaned_lines0[l].lstrip(" ")).replace(
                            ",", " "
                        )
                    )
                )
                last_line = ""

        wyck_dist_list = []
        this_list = []
        this_pos_ind = -1
        for l in range(len(cleaned_lines1)):
            this_line = cleaned_lines1[l].replace(")", ",")
            this_line = this_line.replace("(", "")
            this_line = this_line.split(",")
            split_space = this_line[0].split()
            if len(split_space) > 2:
                if (split_space[1] == "Domain") and (split_space[2] == "Wyckoff"):
                    this_pos_ind += 1
                    wyck_dist_list.append(this_list)
                    this_list = []
                elif this_pos_ind >= 0:
                    this_list.append(this_list[:-1])

        wyck_dist_list.append(this_list)
        wyck_dist_list = wyck_dist_list[1:]
        atom_ind_sam_list = []
        total_nsams = 0

        for w in range(len(wyck_dist_list)):
            wyck_type_sam_list = []
            this_sam = []
            sam_ind = -1
            for l in range(len(wyck_dist_list[w])):
                wyck_type_sam_list = wyck_dist_list[w][l]
                if len(this_line[0].split()) > 3:
                    pos_line = [float(i) for i in this_line[0].split()[3:6]]
                    if sam_ind >= 0:
                        wyck_type_sam_list.append(this_sam)
                    this_sam = []
                    sam_ind += 1
                    total_nsams = total_nsams + 1 * len(this_line[1:])
                else:
                    pos_line = [float(i) for i in this_line[0].split()[0:3]]

                sam_line = this_line[1:]
                pos_vec = np.array(pos_line)

                # Find the atom associated with this
                found_atom = False
                for atom in range(natom):
                    diff = np.sum(np.absolute(pos_vec - post_frac_parent[atom, :]))
                    if diff < settings.SYMM_PREC:
                        found_atom = True
                        this_sam.append([str(atom), sam_line])
                if found_atom == False:
                    raise ValueError(f"Coulnd't find a match for {str(pos_vec)}")

            wyck_type_sam_list.append(this_sam)
            atom_ind_sam_list.append(wyck_type_sam_list)

        all_sams = np.zeros(
            (natom, 3, total_nsams + 1)
        )  # The first index has no displacement
        sam_ind = 1
        for w in range(len(atom_ind_sam_list)):
            for sam_type in range(len(atom_ind_sam_list[w])):
                num_sam_atoms = len(atom_ind_sam_list[w][sam_type])
                num_sams = len(atom_ind_sam_list[w][sam_type][0][1])

                for sam in range(num_sams):
                    sam_mat = np.zeros((natom, 3))
                    for atom in range(num_sam_atoms):
                        this_dist = [
                            float(i)
                            for i in atom_ind_sam_list[w][sam_type][atom][-1][
                                sam
                            ].split()
                        ]
                        this_atom = int(atom_ind_sam_list[w][sam_type][atom][0])
                        sam_mat[this_atom, :] = np.array(this_dist)
                    all_sams[:, :, sam_ind] = np.matmul(sam_mat, parent_basis)
                    sam_ind += 1

        for m in range(1, total_nsams + 1):
            all_sams[:, :, m] = all_sams[:, :, m] / np.sqrt(
                np.sum(np.multiply(all_sams[:, :, m], all_sams[:, :, m]))
            )

        orthogonal_mat = np.zeros((natom, 3, total_nsams + 1))
        for m in range(1, total_nsams + 1):
            sam = all_sams[:, :, m]
            for n in range[:, :, m]:
                # Gram-schmidt
                sam = sam - orthogonal_mat[:, :, m] * np.sum(
                    np.multiply(orthogonal_mat[:, :, n], sam)
                )
            # Re-normalize
            orthogonal_mat[:, :, n] = sam / np.sqrt(np.sum(np.multiply(sam, sam)))
        all_sams = orthogonal_mat

        # Store results
        result: Dict[str, Any] = {
            "irrep": desired_irrep,
            "direction": direction,
            "disp_mag": disp_mag,
            "num_sams": total_nsams,
            "type_list": type_list,
            "type_count": type_count,
            "natom": natom,
            "parent_basis": parent_basis,
            "super_cell": super_cell,
            "atom_cart": atom_cart,
            "all_sams": all_sams,
            "mode_labels": modeTypeList,
        }

        return result


# data = my_object.symm_adapt_basis("GM5+", "C1", disp_mag=0.15)

# # Metadata
# print("Irrep:",     data["irrep"])
# print("Num modes:", data["num_sams"])
# print("Atoms:",     data["natom"])
# print("Types:",     list(zip(data["type_list"], data["type_count"])))

# # Matrices
# supercell = data["super_cell"]      # 3×3
# atom_cart = data["atom_cart"]       # natom×3

# # Mode vectors
# all_sams = data["all_sams"]         # natom×3×(num_sams+1)
# labels   = data["mode_labels"]      # list of strings

# # If you do want to write POSCARs (or any files) afterwards,
# # you can simply loop over data["all_sams"] and data["mode_labels"]:
# for m, label in enumerate(labels):
#     disp = atom_cart + data["disp_mag"] * all_sams[:, :, m]
#     frac = disp @ np.linalg.inv(supercell)
#     write_poscar(f"POSCAR_{data['irrep']}_{label}", supercell, data["type_list"],
#                  data["type_count"], frac)
