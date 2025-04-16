# pipeline.py
import os
from typing import List, Dict, Optional
from symmstate.config.symm_state_settings import SymmStateSettings

from symmstate.slurm_file import SlurmFile  # your SlurmFile class


class Pipe:
    """
    A generic pipeline class that can run multi-stage jobs (Abinit, VASP, etc.)
    using SlurmFile for batch script writing and submission.
    """

    def __init__(self, slurm_file: SlurmFile):
        """
        Args:
            slurm_file: A SlurmFile instance for writing/submitting SLURM jobs.
        """
        self.slurm_file = slurm_file
        self.project_root = SymmStateSettings.PROJECT_ROOT

    def _join_project_path(self, *subdirs: str) -> str:
        """
        Joins subdirectory paths to the PROJECT_ROOT, returning an absolute path.
        """
        return os.path.join(self.project_root, *subdirs)

    def run_stage(
        self,
        stage_name: str,
        executable: str,
        input_file: str,
        log_file: str,
        extra_commands: Optional[str] = None,
    ) -> str:
        """
        Runs a single stage in the pipeline using SLURM for HPC scheduling.

        Args:
            stage_name: A user-friendly name for this stage (e.g. "ground_state_calc").
            executable: The program or command to run (e.g. "abinit", "vasp", "./my_tool").
            input_file: The main input file path (absolute or relative).
            log_file: The file where standard output will be saved.
            extra_commands: Optional shell commands appended to the script after the main command.

        Returns:
            The path to the created SLURM batch script.
        """
        # Make sure logs directory exists
        logs_dir = self._join_project_path("logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Construct absolute path for the log file (if not already)
        abs_log_file = os.path.join(logs_dir, os.path.basename(log_file))

        # Build a batch script name under a "scripts" folder
        scripts_dir = self._join_project_path("scripts")
        os.makedirs(scripts_dir, exist_ok=True)

        batch_script_name = f"{stage_name}_batch.sh"
        batch_script_path = os.path.join(scripts_dir, batch_script_name)

        # The main MPI/command line for SlurmFile:
        # For instance, SlurmFile might have:
        #   mpi_command_template="mpirun -np {num_procs} {executable} < {input_file} > {log}"
        # so we pass "executable" in the "input_file" placeholder, or adapt your template.
        #
        # Alternatively, if your SlurmFile uses a simpler approach, just do:
        #   slurm_file.write_batch_script(input_file=input_file, log_file=abs_log_file, ...)
        # and ensure it knows how to insert "executable".

        # In many HPC setups, you'd incorporate 'executable' into the template or pass it as extra_commands:
        # We'll keep this example simple. Let's say your SlurmFile can handle an "executable" format arg:
        script_created = self.slurm_file.write_batch_script(
            input_file=input_file,  # could be the input file name
            log_file=abs_log_file,  # logs
            batch_name=batch_script_path,
            extra_commands=extra_commands,
        )

        # Submit the job
        self.slurm_file.submit_job(script_created)

        return script_created

    def run_pipeline(self, stages: List[Dict]) -> None:
        """
        Runs multiple stages in sequence, each a dict with:
          - stage_name (str)
          - executable (str)  # e.g. "abinit", "vasp"
          - input_file (str)
          - log_file (str)
          - (optional) extra_commands (str)

        Example:
          stages = [
            {
              "stage_name": "gs_calc",
              "executable": "abinit",
              "input_file": "/abs/path/to/mycell_gs.abi",
              "log_file": "gs_calc.log"
            },
            {
              "stage_name": "piezo_step",
              "executable": "abinit",
              "input_file": "/abs/path/to/mycell_piezo.abi",
              "log_file": "piezo.log",
              "extra_commands": "echo 'Piezo done!'"
            }
          ]
        """
        for stg in stages:
            stage_name = stg["stage_name"]
            exe = stg["executable"]
            input_f = stg["input_file"]
            log_f = stg["log_file"]
            extra_cmd = stg.get("extra_commands", None)

            print(f"--- Running stage: {stage_name} with {exe} ---")
            script_path = self.run_stage(
                stage_name=stage_name,
                executable=exe,
                input_file=input_f,
                log_file=log_f,
                extra_commands=extra_cmd,
            )
            print(f"Stage '{stage_name}' submitted with script: {script_path}")

        # Optionally block until all jobs finish
        self.slurm_file.wait_for_jobs_to_finish(check_time=30)
        print("All stages have completed (or are at least submitted).")
