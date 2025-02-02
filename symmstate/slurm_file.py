from symmstate import SymmStateCore
import os
import warnings
import subprocess
import time


class SlurmFile(SymmStateCore):
    """
    Serves as the class that writes sbatch scripts
    """

    def __init__(self, sbatch_header_source):
        """
        Takes either a multiline string or a file as the header of the batch script
        """
        self.running_jobs = []
        self.batch_header = None
        if os.path.isfile(sbatch_header_source):
            try:
                with open(sbatch_header_source, "r") as file:
                    self.batch_header = file.read()
            except Exception as e:
                raise ValueError(
                    f"An error occurred while reading the header file: {e}"
                )
        else:
            self.batch_header = sbatch_header_source

        print(f"Batch header: {self.batch_header}")

    def write_batch_script(
        self,
        input_file="input.in",
        batch_name="default_output",
        host_spec=None,
        log="log",
    ):
        """
        Writes a batch script using the header from sbatch_header, customizing it for Abinit execution.

        Args:
            input_file (str): Input file name used in Abinit execution. Defaults to 'input.in'.
            batch_name (str): Name of the resulting batch script. Defaults to 'default_output'.
            host_spec (str): Host specification line for distributed computing. Optional.
            log (str): Log file name for capturing output messages. Defaults to 'log'.

        Returns:
            bool: True if the batch script was successfully written; False otherwise.
        """

        # Check if the file already exists
        if os.path.exists(batch_name):
            warnings.warn(
                f"The file '{batch_name}' already exists and will be overwritten.",
                UserWarning,
            )

        # Write to the output file using the header from self.sbatch_header
        try:
            with open(batch_name, "w") as file:
                # Write the contents of the batch script header
                file.write("#!/bin/bash\n")
                file.write(self.batch_header)

                # Write the rest of the batch script
                if host_spec is None:
                    file.write(f"\nmpirun -np 8 abinit < {input_file} > {log} \n")
                else:
                    file.write(f"\n{host_spec} abinit < {input_file} > {log} \n")

            print("Batch script was written successfully.")
            return True

        except Exception as e:
            print(f"An error occurred while writing the batch script: {e}")
            return False

    def all_jobs_finished(self):
        """
        Checks whether all submitted computational jobs have finished.

        Returns:
            bool: True if all jobs have completed; False otherwise.
        """
        for job_id in self.running_jobs:
            # Check the status of the specific job
            result = subprocess.run(
                ["squeue", "-j", str(job_id)], capture_output=True, text=True
            )

            if result.returncode != 0:
                print(f"Error checking job {job_id} status:", result.stderr)
                continue

            # If the job is found in the queue, it means it's still running or pending
            if str(job_id) in result.stdout:
                return False

        # If none of the jobs were found in the queue, they have all finished
        return True

    def wait_for_jobs_to_finish(self, check_time=60):
        """
        Waits until all computational jobs are finished, checking their status periodically.

        Args:
            check_time (int): Time interval between checks, in seconds. Defaults to 60.
        """
        print("Waiting for jobs to finish...")
        while not self.all_jobs_finished():
            print(f"Jobs still running. Checking again in {check_time} seconds...")
            time.sleep(check_time)
        print("All jobs have finished.")
        self.running_jobs = []
