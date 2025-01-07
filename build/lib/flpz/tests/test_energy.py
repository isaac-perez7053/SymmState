import numpy as np
import unittest
from pathlib import Path
from symmstate import Energy

# Path to SMODES executable
SMODES_PATH = "/home/iperez/isobyu/smodes"  # Update this with the actual path to your SMODES executable
INPUT_FILE = "/home/iperez/flpz/tests/example1_inputFile.in"
# Path to SMODES input file and target irreducible representation
SMODES_INPUT = "/home/iperez/flpz/tests/example_1sInput.txt"  # Update this with the actual SMODES input file path
TARGET_IRREP = "GM4-"  # Update this with the target irreducible representation
ABIFILE = "/home/iperez/flpz/tests/example1_file.abi"
B_SCRIPT_HEADER = "/home/iperez/flpz/tests/b-script-preamble.txt"

class TestEnergy(unittest.TestCase):
    def test_input_file_1(self):
        energy_calculation = Energy(INPUT_FILE, SMODES_INPUT, TARGET_IRREP, smodes_path=SMODES_PATH)
        energy_calculation.run_program()


if __name__ == "__main__":
    unittest.main()
