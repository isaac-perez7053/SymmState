from symmstate.flpz.electrotensor import ElectroTensorProgram
from symmstate.slurm import SlurmFile, SlurmHeader

# Path to SMODES executable
SMODES_PATH = "/home/iperez/isobyu/smodes"  # Update this with the actual path to your SMODES executable
INPUT_FILE = "/home/iperez/symmstate_testing/example1_inputFile.in"
# Path to SMODES input file and target irreducible representation
SMODES_INPUT = "/home/iperez/symmstate_testing/example_1sInput.txt"  # Update this with the actual SMODES input file path
TARGET_IRREP = "GM4-"  # Update this with the target irreducible representation
ABIFILE = "/home/iperez/symmstate_testing/example1_file.abi"
B_SCRIPT_HEADER = "/home/iperez/symmstate_testing/b-script-preamble.txt"

header = SlurmHeader(
    job_name="abinit",
    partition="LocalQ",
    time="159:59:59",
    output="abinit.%j.%N.out",
    additional_lines=[
        "#SBATCH --ntasks-per-node=30",
        "#SBATCH --mem=10000M",
        "#SBATCH --export=ALL",
    ],
)

# 2. Create the SlurmFile object
#    num_processors=8 -> used in the MPI template
#    The mpi_command_template includes 'mpirun -hosts=localhost -np {num_procs} ...'
slurm_manager = SlurmFile(
    slurm_header=header,
    num_processors=8,
    mpi_command_template=(
        "mpirun -hosts=localhost -np {num_procs} abinit {input_file} > {log}"
    ),
)
electrotensor_calculation = ElectroTensorProgram(
    name="electrotensor_program_test",
    num_datapoints=2,
    abi_file=ABIFILE,
    min_amp=0.0,
    max_amp=0.7,
    smodes_input=SMODES_INPUT,
    target_irrep=TARGET_IRREP,
    slurm_obj=slurm_manager,
)
electrotensor_calculation.run_program()
