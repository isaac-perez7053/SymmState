from symmstate.flpz.energy import EnergyProgram
from symmstate.slurm_file import SlurmFile

# Path to SMODES executable
SMODES_PATH = "/home/iperez/isobyu/smodes"  # Update this with the actual path to your SMODES executable
INPUT_FILE = "/home/iperez/symmstate_testing/example1_inputFile.in"
# Path to SMODES input file and target irreducible representation
SMODES_INPUT = "/home/iperez/symmstate_testing/example_1sInput.txt"  # Update this with the actual SMODES input file path
TARGET_IRREP = "GM4-"  # Update this with the target irreducible representation
ABIFILE = "/home/iperez/symmstate_testing/example1_file.abi"
B_SCRIPT_HEADER = "/home/iperez/symmstate_testing/b-script-preamble.txt"

slurm_obj = SlurmFile(B_SCRIPT_HEADER, num_processors=8)
energy_calculation = EnergyProgram(name="energy_program_test",
num_datapoints=12, abi_file=ABIFILE, min_amp=0.0, max_amp=0.7, smodes_input=SMODES_INPUT, target_irrep=TARGET_IRREP, slurm_obj=slurm_obj)
energy_calculation.run_program()


