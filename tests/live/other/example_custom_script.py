from symmstate.abinit import AbinitFile
from symmstate.slurm import SlurmFile, SlurmHeader


# Energy analysis (how well is it converged)
def convergence_err(energy_arr):

    convergence_err = []
    for i in range(0, len(energy_arr)):
        energy_pre = energy_arr[i]
        energy_post = energy_arr[i + 1]
        convergence_err.append((energy_pre - energy_post) / energy_post)

    return convergence_err


# 1. Establish Slurm file header

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
slurm_obj = SlurmFile(
    slurm_header=header,
    num_processors=8,
    mpi_command_template=(
        "mpirun -hosts=localhost -np {num_procs} abinit {input_file} > {log}"
    ),
)

abinit_file = AbinitFile(abi_file="test_file.abi")

content: str = """ndtset 2
chkprim 0
kptopt 2

# Set 1 : Ground State Self-Consistent Calculation
#************************************************
  kptopt1 1
  tolvrs 1.0d-18

# Set 2 : Calculation of ddk wavefunctions
#************************************************
  kptopt2 2
  getwfk2 1
  rfelfd2 2
  iscf2   -3
  tolwfr2 1.0D-18
"""

# Write the custom abinit file
output_file = abinit_file.write_custom_abifile(
    output_file="custom_abifile.abi", content=content, coords_are_cartesian=True
)

# Run abinit using the custom abinit file
abinit_file.run_abinit(
    output_file, slurm_obj=slurm_obj, batch_name="custom_batch.sh", log_file="log"
)

# Wait for job to finish
slurm_obj.wait_for_jobs_to_finish()

# Do whatever you'd like with the finished file!
