import numpy as np
from pymatgen.core import Structure, Element
from symmstate.utils.misc import Misc
from symmstate.abinit import AbinitFile


abinit_file = AbinitFile(
    abi_file="test_file.abi",
    smodes_input="example_smodes_input_M3+.txt",
    target_irrep="M3+",
)

print(f"Printing coords: {abinit_file.vars['xcart']}")
print(f"Printing coords: {abinit_file.vars['xred']}")

print(f"Printing typat: {abinit_file.vars['typat']}")

sites = abinit_file.structure.sites
print(sites)


# species = [site.specie for site in sites]
# natom = len(sites)
# # Preserve the original unique order as they appear.
# original_znucl = list(dict.fromkeys([s.Z for s in species]))  # e.g. [20, 8, 22]
# # Original typat follows the original order.
# original_typat = [original_znucl.index(s.Z) + 1 for s in species]
# ntypat = len(original_znucl)

# Calculate the number of bands using an external routine.
# nband = Misc.calculate_nband(structure)
