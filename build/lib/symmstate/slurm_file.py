from symmstate import SymmStateCore
import os
import warnings
import subprocess
import time
from typing import Optional, Union

class SlurmFile(SymmStateCore):
    """
    Manages creation and execution of SLURM batch scripts with enhanced job monitoring.
    """

    def __init__(self, sbatch_header_source: Union[str, os.PathLike], num_processors: int = 8):
        """
        Initialize with batch script header and processor count.

        Args:
            sbatch_header_source: Multiline string or file path containing SLURM header
            num_processors: Default number of MPI processors (default: 8)
        """
        super().__init__()
        self.num_processors = num_processors
        self.running_jobs = []

        # Handle multiline string or file input
        if isinstance(sbatch_header_source, str) and '\n' in sbatch_header_source:
            self.batch_header = sbatch_header_source
        elif os.path.isfile(sbatch_header_source):
            try:
                with open(sbatch_header_source, "r") as file:
                    self.batch_header = file.read()
            except Exception as e:
                raise ValueError(f"Error reading header file: {e}")
        else:
            self.batch_header = sbatch_header_source

        print(f"Initialized SLURM manager with {self.num_processors} processors")

    def write_batch_script(
        self,
        input_file: str = "input.in",
        log_file: str = "job.log",
        batch_name: str = "job.sh",
        mpi_command_template: str = "mpirun -np {num_procs} abinit < {input_file} > {log}",
        extra_commands: Optional[str] = None,
    ) -> str:
        """
        Write a SLURM batch script with a customizable MPI execution line.
        
        Args:
            input_file: Name of the input file for the calculation.
            log_file: Name of the file to store output logs.
            batch_name: Name of the batch script to write.
            mpi_command_template: Template for the MPI command line.
            extra_commands: Optional string of additional shell commands to insert after the MPI line.
            
        Returns:
            str: The path to the written batch script.
        """
        # Define the shebang line (for bash)
        shebang = "#!/bin/bash\n"
        
        mpi_line = mpi_command_template.format(
            num_procs=self.num_processors,
            input_file=input_file,
            log=log_file
        )
        
        # Prepend the shebang and the existing header to create the final script content.
        script_content = f"{shebang}{self.batch_header.strip()}\n\n{mpi_line}"
        if extra_commands:
            script_content += f"\n\n{extra_commands.strip()}\n"
        
        with open(batch_name, "w") as script_file:
            script_file.write(script_content)
        
        print(f"Wrote batch script to {batch_name}")
        return batch_name



    def all_jobs_finished(self) -> bool:
        """
        Comprehensive job completion check with error handling.
        
        Returns:
            bool: True if all jobs have successfully completed, False otherwise
        """
        if not self.running_jobs:
            return True

        completed_jobs = []
        all_finished = True

        for job_id in self.running_jobs:
            try:
                # Check job status using sacct for completed jobs and squeue for active ones
                result = subprocess.run(
                    ["sacct", "-j", str(job_id), "--format=State"],
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                
                # Parse job state
                state_lines = [line.strip() for line in result.stdout.split('\n')[2:] if line.strip()]
                job_states = [line.split()[0] for line in state_lines if line]
                
                if not job_states:
                    # Fallback to squeue check if sacct fails
                    result = subprocess.run(
                        ["squeue", "-j", str(job_id)], 
                        capture_output=True, 
                        text=True,
                        timeout=10
                    )
                    if str(job_id) in result.stdout:
                        all_finished = False
                    else:
                        completed_jobs.append(job_id)
                else:
                    # Check for completion states
                    if all(state in ['COMPLETED', 'CANCELLED', 'FAILED'] for state in job_states):
                        completed_jobs.append(job_id)
                    else:
                        all_finished = False

            except subprocess.TimeoutExpired:
                print(f"Timeout checking job {job_id} status")
                all_finished = False
            except Exception as e:
                print(f"Error checking job {job_id}: {str(e)}")
                all_finished = False

        # Remove completed jobs from tracking
        self.running_jobs = [j for j in self.running_jobs if j not in completed_jobs]

        return all_finished and not self.running_jobs

    def wait_for_jobs_to_finish(self, check_time: int = 60, check_once: bool = False) -> None:
        """
        Robust waiting mechanism with completion state verification.
        
        Args:
            check_time: Polling interval in seconds (default: 60)
            check_once: Single check flag for testing (default: False)
        """
        print(f"Monitoring {len(self.running_jobs)} jobs...")
        try:
            if check_once:
                time.sleep(check_time)
                self.all_jobs_finished()
            else:
                while True:
                    if self.all_jobs_finished():
                        break
                    print(f"Jobs remaining: {len(self.running_jobs)} - Next check in {check_time}s")
                    time.sleep(check_time)
        except KeyboardInterrupt:
            print("\nJob monitoring interrupted by user")
        finally:
            if self.running_jobs:
                print(f"Warning: {len(self.running_jobs)} jobs still tracked after monitoring")
            else:
                print("All jobs completed successfully")

    # Rest of the class remains unchanged (write_batch_script, etc.)

# Example use:

## Use a custom command template
# slurm.write_batch_script(
#     mpi_command_template="srun -n {num_procs} --mpi=pmi2 abinit < {input_file} | tee {log}"
# )

# # Custom template with host specification
# slurm.write_batch_script(
#     host_spec="mpiexec -hostfile nodefile",
#     mpi_command_template="{host_spec} -np {num_procs} abinit > {log}"
#)

## Create with multiline header
# header = """#SBATCH --nodes=2
# #SBATCH --ntasks-per-node=16
# #SBATCH --time=01:00:00"""

# slurm = SlurmFile(header, num_processors=32)

# # Write batch script with custom processors
# slurm.write_batch_script(
#     batch_name="job.sh",
#     mpi_command_template="mpirun -np {num_procs} abinit < {input_file} > {log}"
# )

# # Monitor jobs with improved tracking
# slurm.running_jobs = [12345, 67890]  # Typically set via job submission
# slurm.wait_for_jobs_to_finish(check_time=300)  # Check every 5 minutes
