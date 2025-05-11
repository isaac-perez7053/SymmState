"""
In short, this test file will submit 12 self-consistent runs that converge the lattice parameters
with respect to different uniform grids and different shiftks
"""

from symmstate.abinit import AbinitFile
from symmstate.slurm import SlurmFile, SlurmHeader
from symmstate.utils import DataParser
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------IGNORE---------------------------------#
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
            error = float("inf")
        else:
            error = (energy_pre - energy_post) / energy_post
        errors.append(error)
    return errors


def convergence_lol(acell_arr):
    """
    Calculate the convergence errors between successive acell values across many
    acells

    Parameters
    ----------
    acell_arr: List of lists
      A list of lists of acell

    Returns
    -------
    errors: List
      A list of lists representing the error associated with each
    """
    errors = []
    L = len(acell_arr[0])
    for i in range(0, L):
        errors.append(convergence_err([err[i] for err in acell_arr]))
    return errors


# -----------------------------------IGNORE---------------------------------#


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


# Storing the list of ngkpt and shiftk I will be testing over
all_output_files = []
shiftk_list = [[0.5, 0.5, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
ngkpt_list = [[2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]]
content = f"""
    Chksymbreak 0
    #Definition of occupation numbers
    occopt 4
    tsmear 0.05

    #Optimization of the lattice parameters
    optcell 1
    ionmov  2
    ntime  10
    dilatmx 1.05
    """

# Choose number of processors you want to use
slurm_obj.num_processors = 12
for idx, shiftk in enumerate(shiftk_list):
    # Choose a single shiftk
    abinit_file.vars["shiftk"] = shiftk
    output_files = []
    for i in range(0, 3):
        abinit_file.vars["ngkpt"] = ngkpt_list[i]

        # 4. Write abinit file and run the job
        output_abi = abinit_file.write_custom_abifile(
            f"lattice_convergence_{i}_{idx}.abi", content, coords_are_cartesian=False
        )
        output_files.append(
            abinit_file.run_abinit(
                input_file=output_abi,
                slurm_obj=slurm_obj,
                log_file=f"convergence_{i}.log",
            )
        )
    # Append all abinit output files for shiftk into an array
    all_output_files.append(output_files)

# -------------------------OPTIONAL----------------------------------#

# Once all jobs are submitted, wait for all of them to finish
slurm_obj.wait_for_jobs_to_finish(check_time=60)


# 5. Extract the second acell from each output file
all_lattice_constants = []
for output_files_list in all_output_files:
    # Choose a single shiftk list
    for output_file in output_files_list:
        lattice_constants = []
        with open(f"{output_file}.abo") as f:
            content = f.read()

        # Extract all instances of the acell keyword in the abo file
        acells = DataParser.parse_array(content, "acell", float, all_matches=True) or []

        if len(acells) < 2:
            raise ValueError(f"{output_file}.abo has only {len(acells)} acell values")

        # Choose the last acell in the file (the converged acell)
        lattice_constants.append(acells[1])

    # Optional: convert to NumPy array for downstream functions
    lattice_array = np.array(lattice_constants)  # shape (n_runs,)
    all_lattice_constants.append(lattice_array)


# Print convergence info
print(convergence_lol(lattice_array))

# 6. Plot the results on a single plot
fig, ax = plt.subplots(figsize=(8, 5))
grid = [2, 4, 6, 8][: len(lattice_constants)]
for idx, lattice_array in enumerate(all_lattice_constants):
    for i in range(0, 2):
        ax.plot(
            grid,
            [latconst[i] for latconst in lattice_array],
            marker="o",
            markersize=8,
            linewidth=1.5,
            label=f"shiftk {shiftk_list[idx]}",
        )
ax.set(
    xlabel="ngkpt",
    ylabel="lattice constant (Bohr)",
    title="ngkpt vs. lattice constants",
)
ax.grid(True, alpha=0.3)
ax.legend()

# Save the plot with a specified DPI and tight layout
fig.savefig("lattice_convergence2.png", dpi=300, bbox_inches="tight")
plt.show()
