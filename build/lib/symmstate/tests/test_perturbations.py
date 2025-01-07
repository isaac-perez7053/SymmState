from pathlib import Path
import numpy as np
import unittest
import ast
from symmstate.abinit import *


# Path to SMODES executable
SMODES_PATH = "/home/iperez/isobyu/smodes"  # Update this with the actual path to your SMODES executable
INPUT_FILE = "/home/iperez/testing/tests/example1_inputFile.in"
# Path to SMODES input file and target irreducible representation
SMODES_INPUT = "/home/iperez/testing/tests/example_1sInput.txt"  # Update this with the actual SMODES input file path
TARGET_IRREP = "GM4-"  # Update this with the target irreducible representation
ABIFILE = "/home/iperez/testing/tests/example1_file.abi"
B_SCRIPT_HEADER = "/home/iperez/testing/tests/b-script-preamble.txt"
min_amp = 0.0
max_amp = 0.5
pert = """[[ 0.4065534792,  0.        ,  0.        ],
 [ 0.3922410056,  0.        ,  0.        ],
 [ 0.8215040921,  0.        ,  0.        ],
 [-0.0547370476,  0.        ,  0.        ],
 [-0.0547370476,  0.        ,  0.        ]]"""
num_datapoints = 11

# Safely evaluate the string to convert it into a list
pert_list = ast.literal_eval(pert)
# Convert the list to a numpy array
pert_array = np.array(pert_list)


class TestPerturbations(unittest.TestCase):
    def test_input_file_1(self):
        perturbations = Perturbations(
            abinit_file=ABIFILE,
            min_amp=min_amp,
            max_amp=max_amp,
            perturbation=pert_array,  # Use the converted numpy array
            batch_script_header_file=B_SCRIPT_HEADER,
        )
        print(f"Perturbation object successfully initialized")
        perturbations.generate_perturbations(num_datapoints=num_datapoints)
        perturbations.calculate_energy_of_perturbations()
        print(f"Energy list: \n \n {perturbations.list_energies} \n")
        print(f"x_vec:\n \n {perturbations.list_amps} \n")
        perturbations.data_analysis(save_plot=True)
        # It seems that I have either extracted the wrong phonon mode, which I can just look through them all and see if I got the right one, or they were incorrectly caclulated.


if __name__ == "__main__":
    unittest.main()
