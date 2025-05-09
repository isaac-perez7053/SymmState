import subprocess
import time
from typing import Optional
from symmstate.slurm.slurm_header import SlurmHeader
import logging
from symmstate._symm_state_core import SymmStateCore


class SlurmFile:
    """
    Manage creation, submission, and monitoring of SLURM batch scripts.

    This class handles the generation of SLURM batch scripts using either a structured
    SlurmHeader or a raw header string, submits jobs via SLURM, and tracks running job IDs.
    It also provides a default MPI command template that can be customized.
    """

    def __init__(
        self,
        slurm_header: Optional[SlurmHeader] = None,
        raw_header: Optional[str] = None,
        num_processors: int = 8,
        mpi_command_template: str = "mpirun -np {num_procs} abinit < {input_file} > {log}",
        logger: logging = None,
    ):
        """
        Initialize a SlurmFile instance.

        Parameters:
            slurm_header (Optional[SlurmHeader]):
                An instance of SlurmHeader containing structured SLURM directives.
            raw_header (Optional[str]):
                A raw, multiline SLURM header to override slurm_header if provided.
            num_processors (int):
                The default number of MPI processors (default is 8).
            mpi_command_template (str):
                A template for constructing the MPI command line used in the batch script.
            logger (logging):
                A logger to information regarding the SlurmFile object
        """
        self.slurm_header = slurm_header
        self.raw_header = raw_header
        self.num_processors = num_processors
        self.mpi_command_template = mpi_command_template
        self._logger = logger

        # We track job IDs for monitoring.
        self.running_jobs = []

        # In case user passes both slurm_header and raw_header, we default to raw_header
        if self.raw_header and self.slurm_header:
            (self._logger.info if self._logger is not None else print)(
                "Warning: raw_header provided; ignoring slurm_header."
            )

        (self._logger.info if self._logger is not None else print)(
            f"Initialized SlurmFile with {self.num_processors} processors"
        )

    def write_batch_script(
        self,
        input_file: str = "input.in",
        log_file: str = "slurm_job.log",
        batch_name: str = "slurm_job.sh",
        extra_commands: Optional[str] = None,
    ) -> str:
        """
        Write a SLURM batch script with the specified input and logging details.

        This function assembles a batch script by combining a shebang, header text (from either
        a raw header or a SLURM header), and an MPI command formatted with the given input and log
        file names. It also appends any extra shell commands if provided, writes the script to disk,
        and returns the script's file path.

        Parameters:
            input_file (str):
                Name of the simulation input file.
            log_file (str):
                Name of the file for standard output.
            batch_name (str):
                Name of the SLURM batch script file (default is "job.sh").
            extra_commands (Optional[str]):
                Additional shell commands to append at the end of the script, if any.

        Returns:
            str: The file path of the written batch script.
        """
        # Standard shebang
        shebang = "#!/bin/bash\n"

        # Derive the header from either raw_header or slurm_header
        if self.raw_header:
            header_text = self.raw_header.strip()
        elif self.slurm_header:
            header_text = self.slurm_header.to_string()
        else:
            header_text = ""  # Possibly no header lines

        # Format the MPI command with placeholders
        mpi_line = self.mpi_command_template.format(
            num_procs=self.num_processors, input_file=input_file, log=log_file
        )

        # Assemble the script
        script_content = f"{shebang}{header_text}\n\n{mpi_line}"
        if extra_commands:
            script_content += f"\n\n{extra_commands}\n"

        # Write the script to disk
        batch_name = SymmStateCore._get_unique_filename(batch_name)
        with open(batch_name, "w") as script_file:
            script_file.write(script_content)

        # Add a logger here
        (self._logger.info if self._logger is not None else print)(
            f"Wrote batch script to {batch_name}"
        )
        return batch_name

    def submit_job(self, batch_script: str) -> Optional[str]:
        """
        Submit a SLURM batch script using 'sbatch' and return the job ID.

        This function runs the 'sbatch' command on the provided batch script and parses the
        output to extract the job ID. If the submission is successful and the job ID is found,
        it is appended to the list of running jobs and returned. If the job ID cannot be parsed
        or an error occurs during submission, the function returns None.

        Parameters:
            batch_script (str):
                The file path to the SLURM batch script to be submitted.

        Returns:
            Optional[str]: The job ID if the submission is successful, otherwise None.
        """
        try:
            result = subprocess.run(
                ["sbatch", batch_script],
                capture_output=True,
                text=True,
                check=True,  # raises CalledProcessError if exit code != 0
            )
            # Typically, SLURM responds with something like: "Submitted batch job 123456"
            output = result.stdout.strip()
            (self._logger.info if self._logger is not None else print)(
                f"sbatch output: {output}"
            )

            # Parse job ID
            # Example: "Submitted batch job 123456"
            if "Submitted batch job" in output:
                job_id = output.split()[-1]
                self.running_jobs.append(job_id)
                (self._logger.info if self._logger is not None else print)(
                    f"Job submitted with ID {job_id}"
                )
                return job_id
            else:
                (self._logger.info if self._logger is not None else print)(
                    "Could not parse job ID from sbatch output."
                )
                return None

        except subprocess.CalledProcessError as e:
            (self._logger.info if self._logger is not None else print)(
                f"Error submitting job: {e.stderr}"
            )
            return None

    def all_jobs_finished(self) -> bool:
        """
        Checks the status of tracked jobs using 'sacct' (and 'squeue' fallback).

        Returns:
            True if all tracked jobs have completed/failed/cancelled, else False.
        """
        if not self.running_jobs:
            return True  # No jobs in the queue

        completed_jobs = []
        all_finished = True

        for job_id in self.running_jobs:
            try:
                # Query sacct for job state
                result = subprocess.run(
                    ["sacct", "-j", str(job_id), "--format=State"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                # Parse out job states
                # sacct output often has header lines, so skip them.
                lines = [
                    ln.strip() for ln in result.stdout.split("\n")[2:] if ln.strip()
                ]
                states = [
                    ln.split()[0] for ln in lines if ln
                ]  # e.g. ['COMPLETED'] or ['RUNNING'] etc.

                if not states:
                    # If sacct didn't find a record, try squeue as fallback
                    sq_result = subprocess.run(
                        ["squeue", "-j", str(job_id)],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if str(job_id) in sq_result.stdout:
                        # It's still running
                        all_finished = False
                    else:
                        # If it's not in squeue either, treat as completed or unknown
                        completed_jobs.append(job_id)
                else:
                    # If we have states, check if they are terminal (COMPLETED, FAILED, CANCELLED, etc.)
                    terminal_states = ["COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"]
                    if all(s in terminal_states for s in states):
                        completed_jobs.append(job_id)
                    else:
                        all_finished = False

            except subprocess.TimeoutExpired:
                (self._logger.info if self._logger is not None else print)(
                    f"Timeout checking job {job_id} status"
                )
                all_finished = False
            except Exception as e:
                (self._logger.info if self._logger is not None else print)(
                    f"Error checking job {job_id}: {str(e)}"
                )
                all_finished = False

        # Remove completed jobs from tracking
        self.running_jobs = [j for j in self.running_jobs if j not in completed_jobs]

        return all_finished and len(self.running_jobs) == 0

    def wait_for_jobs_to_finish(
        self, check_time: int = 60, check_once: bool = False
    ) -> None:
        """
        Poll job statuses until all jobs are finished or monitoring is interrupted.

        This function repeatedly checks the status of running jobs, pausing for 'check_time' seconds
        between checks. If 'check_once' is True, it performs a single check (useful for testing).
        The process continues until all tracked jobs have completed or the user interrupts via Ctrl+C.

        Parameters:
            check_time (int): Time in seconds to wait between job status checks (default is 60).
            check_once (bool): If True, perform only a single check instead of continuous polling.
        """
        (self._logger.info if self._logger is not None else print)(
            f"Monitoring {len(self.running_jobs)} jobs..."
        )
        try:
            if check_once:
                time.sleep(check_time)
                self.all_jobs_finished()
            else:
                while True:
                    if self.all_jobs_finished():
                        break
                    (self._logger.info if self._logger is not None else print)(
                        f"Jobs remaining: {len(self.running_jobs)} - next check in {check_time}s"
                    )
                    time.sleep(check_time)
        except KeyboardInterrupt:
            (self._logger.info if self._logger is not None else print)(
                "\nJob monitoring interrupted by user!"
            )
        finally:
            if self.running_jobs:
                (self._logger.info if self._logger is not None else print)(
                    f"Warning: {len(self.running_jobs)} jobs still tracked after monitoring."
                )
            else:
                (self._logger.info if self._logger is not None else print)(
                    "All jobs completed successfully!"
                )

    def set_logger(self, logger: logging) -> None:
        self._logger = logger
