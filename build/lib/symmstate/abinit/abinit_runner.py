import os
from typing import Optional

# Suppose you import your path utilities here
from symmstate.utils import get_data_path, get_scripts_path, get_logs_path, get_temp_path
from symmstate.slurm import SlurmFile
from abinit_file import AbinitFile  # your custom class providing _get_unique_filename()

class MyAbinitRunner:

    def run_abinit(
        self,
        input_file: str,
        slurm_obj: Optional[SlurmFile] = None,
        *,
        batch_name: Optional[str],
        log_file: Optional[str],
        extra_commands: Optional[str],
    ) -> None:
        """
        Executes the Abinit program using a generated input file and specified settings.
        """
        # For illustration, we build a small 'content' string
        content: str = f"""{input_file}.abi
{input_file}.abo
{input_file}o
{input_file}_gen_output
{input_file}_temp
        """

        # Example: create a unique data file path (absolute) in a 'temp' or 'data' folder:
        file_path = f"{input_file}_abinit_input_data.txt"
        # Use your unique-filename logic:
        file_path = AbinitFile._get_unique_filename(file_path) 
        # Then convert it to an absolute path:
        abs_file_path = get_temp_path(os.path.basename(file_path))  
        # or get_data_path(...) if you prefer storing in "data"

        # Write the content file
        with open(abs_file_path, "w") as file:
            file.write(content)

        if slurm_obj is not None:
            try:
                # Adjust the batch name to a unique path under 'scripts/'
                unique_sh_name = AbinitFile._get_unique_filename(f"{batch_name}.sh")
                abs_batch_name = get_scripts_path(os.path.basename(unique_sh_name))

                # Adjust the log file to an absolute path, e.g. in 'logs/'
                abs_log_file = get_logs_path(log_file)

                # Write the batch script via SlurmFile
                script_created = slurm_obj.write_batch_script(
                    input_file=f"{input_file}.abi",  # or an absolute path if needed
                    log_file=abs_log_file,
                    batch_name=abs_batch_name,
                    extra_commands=extra_commands
                )

                self._logger.info(f"Batch script created: {script_created}")

                # Submit the job
                slurm_obj.submit_job(script_created)

            finally:
                # Your code always raises here—maybe that’s intentional for debugging?
                raise RuntimeError("Failed to run abinit using the batch script!") 

        else:
            # If no SlurmFile provided, run abinit directly in local environment
            abs_log_file = get_logs_path(log_file)
            command: str = f"abinit {input_file} > {abs_log_file}"
            os.system(command)
            self._logger.info(f"Abinit executed directly. Output written to '{abs_log_file}'.")
