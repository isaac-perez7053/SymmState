from symmstate.abinit import SmodesProcessor
import unittest
import numpy as np

# Path to SMODES executable
SMODES_PATH = "/home/iperez/isobyu/smodes"  # Update this with the actual path to your SMODES executable
INPUT_FILE = "/home/iperez/testing/tests/example1_inputFile.in"
# Path to SMODES input file and target irreducible representation
SMODES_INPUT = "/home/iperez/testing/tests/example_1sInput.txt"  # Update this with the actual SMODES input file path
TARGET_IRREP = "GM4-"  # Update this with the target irreducible representation
ABIFILE = "/home/iperez/testing/tests/example1_file.abi"
B_SCRIPT_HEADER = "/home/iperez/testing/tests/b-script-preamble.txt"


class TestAbinitFile(unittest.TestCase):
    # Instantiate the processor
    processor = SmodesProcessor(
        abi_file=ABIFILE,
        target_irrep=TARGET_IRREP,
        smodes_input=SMODES_INPUT,
        b_script_header_file=B_SCRIPT_HEADER,
        disp_mag=0.001,
        smodes_path=SMODES_PATH,
    )

    # # Run the processing pipeline
    # processor.process(SMODES_INPUT)

    # # Print results for verification
    # print("Force Matrix:\n", processor.force_matrix)
    # print("\nMass Matrix:\n", processor.mass_matrix)
    # print("\nDynamical Matrix:\n", processor.dynamical_matrix)
    # print("\nFrequencies (THz):\n", processor.frequencies_thz)
    # print("\nFrequencies (cm^-1):\n", processor.frequencies_cm)
    print("initialization complete")

    processor.symmadapt()
    print("symmadapt complete")
    print(f"fc_evals:\n {processor.fc_evals} \n")
    print(f"phonon_vecs: \n {processor.phonon_vecs} \n")
    print(f"dyn_freqs: \n {processor.dyn_freqs} \n")
    unstable_phonons = np.array(processor.unstable_phonons())
    print(f"unstable phonons: \n {unstable_phonons} \n")


if __name__ == "__main__":
    unittest.main()
