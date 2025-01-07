from symmstate import SymmStateCore
import os
import warnings


class SlurmFile(SymmStateCore):
    """
    Serves as the class that writes sbatch scripts
    """

    def __init__(self, sbatch_header_source):
        """
        Takes either a multiline string or a file as the header of the batch script
        """
        self.batch_header = None
        if os.path.isfile(sbatch_header_source):
            try:
                with open(sbatch_header_source, "r") as file:
                    self.sbatch_header = file.read()
            except Exception as e:
                raise ValueError(
                    f"An error occurred while reading the header file: {e}"
                )
        else:
            self.sbatch_header = sbatch_header_source

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
                file.write(self.sbatch_header)

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
