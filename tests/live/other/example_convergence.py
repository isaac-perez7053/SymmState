"""
In short, this test file will submit 11 energy calculations with ecuts form 10 - 37 
and compute/plot the convergence of the Abinit system using SymmState
"""

from symmstate.abinit import AbinitFile
from symmstate.slurm import SlurmFile, SlurmHeader
from symmstate.utils import DataParser
import matplotlib.pyplot as plt


def convergence_err(energy_arr):
    """
    Calculate convergence errors between successive energy values.

    This function computes the convergence error for each successive pair of energy 
    values in energy_arr using the formula:

        error_i = (E[i] - E[i+1]) / E[i+1]

    Parameters:
        energy_arr (list or array-like):
            A sequence of energy values.

    Returns:
        list:
            A list containing the convergence error for each consecutive pair of energy values.
    """
    errors = []
    for i in range(len(energy_arr) - 1):
        energy_pre = energy_arr[i]
        energy_post = energy_arr[i + 1]
        # Guard against division by zero
        if energy_post == 0:
            error = float('inf')
        else:
            error = (energy_pre - energy_post) / energy_post
        errors.append(error)
    return errors


# 1. Establish Slurm file header

header = SlurmHeader(
    job_name="abinit",
    partition="LocalQ",
    time="159:59:59",
    output="abinit.%j.%N.out",
    additional_lines=[
        "#SBATCH --ntasks-per-node=12",
        "#SBATCH --mem=10000M",
        "#SBATCH --export=ALL",
    ],
)

# 2. Create the SlurmFile object
slurm_obj = SlurmFile(
    slurm_header=header,
    num_processors=12,
    mpi_command_template=(
        "mpirun -hosts=localhost -np {num_procs} abinit {input_file} > {log}"
    ),
)

# 3. Initialize AbinitFile
abinit_file = AbinitFile("test_file.abi")
energy = []
ecuts = []

# Set starting value of ecut. Here, I just set it to 10 Ha
abinit_file.vars["ecut"] = 10
ecuts.append(10)
output_files = []

# Unperturbed energy calculation
# run_energy_calculation outputs the output_file name
output_files.append(
    abinit_file.run_energy_calculation(slurm_obj=slurm_obj, log_file="energy_log")
)

for i in range(1, 10):

    # Iterate ecut by factors of 3. Therefore, we will be measuring the evolution of the energy
    # between ecuts of 10 - 37
    abinit_file.vars["ecut"] += 3
    ecuts.append(abinit_file.vars["ecut"])
    output_files.append(
        abinit_file.run_energy_calculation(slurm_obj=slurm_obj, log_file="energy_log")
    )

# Once all jobs are submitted, wait for all of them to finish
slurm_obj.wait_for_jobs_to_finish(check_time=60)

# Extract all energies from finished calculations
for output_file in output_files:
    energy.append(
        DataParser.grab_energy(
            abo_file=f"{output_file}.abo", logger=abinit_file._logger
        )
    )

# Print convergence_arr array
print(convergence_err(energy))

# Plot results and save plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ecuts, energy, marker='o', markersize=8, linewidth=1.5)
ax.set(xlabel="ecut (Ha)", ylabel="Energy (Ha)", title="Energy vs Ecut")
ax.grid(True, alpha=0.3)

# Save the plot with a specified DPI and tight layout
fig.savefig("energy_vs_ecut.png", dpi=300, bbox_inches="tight")
plt.show()
