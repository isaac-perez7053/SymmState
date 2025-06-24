from symmstate.flpz.energy import EnergyProgram
from symmstate.slurm import SlurmFile, SlurmHeader
import numpy as np

# Path to SMODES input file and target irreducible representation
SMODES_INPUT = "/home/iperez/symmstate_testing/example_1sInput.txt"  # Update this with the actual SMODES input file path
TARGET_IRREP = "GM4-"  # Update this with the target irreducible representation
ABIFILE = "/home/iperez/symmstate_testing/example1_file.abi"

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
slurm_manager = SlurmFile(
    slurm_header=header,
    num_processors=8,
    mpi_command_template=(
        "mpirun -hosts=localhost -np {num_procs} abinit {input_file} > {log}"
    ),
)

perturbation= np.array([[-3.8175 ,       1.2725  ,      6.3625      ],
 [ 1.2721142614 , 6.3621142614 ,16.5421142614],
 [ 6.3625     ,  11.4525     ,  26.7225      ],
 [11.4528857386, 16.5428857386, 36.9028857386],
 [ 1.2727727597 , 1.2727727597, 11.4527727597],
 [ 6.3627727597 , 6.3627727597, 21.6327727597],
 [11.4522272403, 11.4522272403, 31.8122272403],
 [16.5422272403, 16.5422272403 ,41.9922272403],
 [19.0875  ,     19.0875  ,     39.4475      ],
 [ 3.8171142614 , 3.8171142614 , 8.9071142614],
 [ 8.9075    ,    8.9075  ,     19.0875      ],
 [13.9978857386 ,13.9978857386, 29.2678857386],
 [-1.2722272403 , 3.8177727597 , 3.8177727597],
 [ 3.8177727597 , 8.9077727597, 13.9977727597],
 [ 8.9072272403, 13.9972272403 ,24.1772272403],
 [13.9972272403 ,19.0872272403 ,34.3572272403]]) * 0.1

energy_calculation = EnergyProgram(
    name="energy_program_test",
    num_datapoints=12,
    abi_file=ABIFILE,
    min_amp=0.0,
    max_amp=0.7,
    smodes_input=SMODES_INPUT,
    target_irrep=TARGET_IRREP,
    slurm_obj=slurm_manager,
)
energy_calculation.run_program()
