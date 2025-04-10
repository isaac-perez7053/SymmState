from pymatgen.core.periodic_table import Element
from pymatgen.core import Structure, Lattice, Element

class Misc:
    def __init__(self):
        pass
    
    @staticmethod
    def calculate_nband(structure: Structure) -> None:
        """
        Calculates and updates the number of bands (nband) in self.vars based on 
        the total number of valence electrons in the current structure.
        
        The procedure is as follows:
        1. For each atom in self.structure, sum up the valence electrons 
            using a predefined mapping.).
        3. Update self.vars["nband"] with the computed value.
        
        Returns:
            int: The calculated number of bands.
        """
        # Simple mapping for common elements.
        valence_map = {
            "H": 1,
            "He": 0,
            "Li": 1,
            "Be": 2,
            "B": 3,
            "C": 4,
            "N": 5,
            "O": 6,
            "F": 7,
            "Ne": 8,
            "Na": 1,
            "Mg": 2,
            "Al": 3,
            "Si": 4,
            "P": 5,
            "S": 6,
            "Cl": 7,
            "Ar": 8,
            "K": 1,
            "Ca": 2,
            "Sc": 3,
            "Ti": 4,
            "V": 5,
            "Cr": 6,
            "Mn": 7,
            "Fe": 8,
            "Co": 9,
            "Ni": 10,
            "Cu": 11,
            "Zn": 12,
            "Ga": 3,
            "Ge": 4,
            "As": 5,
            "Se": 6,
            "Br": 7,
            "Kr": 8,
            "Rb": 1,
            "Sr": 2,
            "Y": 3,
            "Zr": 4,
            "Nb": 5,
            "Mo": 6,
            "Tc": 7,
            "Ru": 8,
            "Rh": 9,
            "Pd": 10,
            "Ag": 11,
            "Cd": 12,
            "In": 3,
            "Sn": 4,
            "Sb": 5,
            "Te": 6,
            "I": 7,
            "Xe": 8,
            "Cs": 1,
            "Ba": 2,
            "La": 3,
            "Ce": 4,   # Depending on oxidation, Ce may be 3 or 4; here we choose 4.
            "Pr": 4,
            "Nd": 4,
            "Pm": 4,
            "Sm": 4,
            "Eu": 4,   # Eu is often considered divalent, but for valence we choose 4 as an approximation.
            "Gd": 4,
            "Tb": 4,
            "Dy": 4,
            "Ho": 4,
            "Er": 4,
            "Tm": 4,
            "Yb": 2,   # Yb is often divalent.
            "Lu": 3,
            "Hf": 4,
            "Ta": 5,
            "W": 6,
            "Re": 7,
            "Os": 8,
            "Ir": 9,
            "Pt": 10,
            "Au": 11,
            "Hg": 12,
            "Tl": 3,
            "Pb": 4,
            "Bi": 5,
            "Po": 6,
            "At": 7,
            "Rn": 8,
            "Fr": 1,
            "Ra": 2,
            "Ac": 3,
            "Th": 4,
            "Pa": 5,
            "U": 6,
            "Np": 7,
            "Pu": 8,
            "Am": 9,
            "Cm": 10,
            "Bk": 11,
            "Cf": 12,
            "Es": 13,
            "Fm": 14,
            "Md": 15,
            "No": 16,
            "Lr": 17,
            "Rf": 4,  # For the transactinides, values are somewhat arbitrary.
            "Db": 5,
            "Sg": 6,
            "Bh": 7,
            "Hs": 8,
            "Mt": 9,
            "Ds": 10,
            "Rg": 11,
            "Cn": 12,
            "Nh": 13,
            "Fl": 14,
            "Mc": 15,
            "Lv": 16,
            "Ts": 17,
            "Og": 18,
        }

        total_valence = 0
        # Loop over all atomic sites in the current structure.
        for specie in structure.species:
            symbol = specie.symbol
            if symbol in valence_map:
                total_valence += valence_map[symbol]
            else:
                # Fallback: use half the atomic number (this is a crude estimate)
                total_valence += specie.Z 
                
        nband = total_valence
        # Figure out why I have to add this 4
        return nband + 4
