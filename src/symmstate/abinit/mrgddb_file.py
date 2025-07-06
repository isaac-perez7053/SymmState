"""
This class is used to interact the mrgddb utility part of Abinit
"""

import os
import subprocess
from symmstate.utils import get_unique_filename
from typing import List
import time


class MrgddbFile:
    """
    Class to interact with the mrgddb utility from Abinit.
    """

    def __init__(self):
        pass

    @staticmethod
    def write(
        input_name: str,
        ddb_files: List[str],
        output_file: str = "out.ddb",
        description: str = None,
    ) -> str:
        """
        Write the mrgddb.in file with the list of DDB files and output DDB name.
        """
        input_name = get_unique_filename(input_name)
        output_file = get_unique_filename(output_file)
        lines = [output_file]
        lines += [description if description is not None else "Description"]
        lines += [str(len(ddb_files))]
        lines += [os.path.abspath(ddb) for ddb in ddb_files]
        lines.append(output_file)
        with open(input_name, "w") as f:
            f.write("\n".join(lines) + "\n")

        return output_file

    @staticmethod
    def execute(input_file: str, log_name: str = "mrgddb.log", sleep_time=10):
        """
        Run the mrgddb utility.
        """
        command = f"mrgddb < {input_file} > {log_name}"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"An error occurred while executing the command: {e}")
        time.sleep(sleep_time)

    # def __repr__(self):
    #     def boxed(title, content):
    #         lines = content.splitlines() if content else []
    #         width = max([len(title)] + [len(line) for line in lines]) + 4
    #         top = f"┌{'─' * (width - 2)}┐"
    #         mid = f"│ {title.ljust(width - 3)}│"
    #         body = "\n".join(f"│ {line.ljust(width - 3)}│" for line in lines) if lines else f"│ {'(empty)'.ljust(width - 3)}│"
    #         bottom = f"└{'─' * (width - 2)}┘"
    #         return f"{top}\n{mid}\n{body}\n{bottom}"

    #     # mrgddb.in content
    #     if os.path.isfile(self.in_path):
    #         try:
    #             with open(self.in_path, "r") as f:
    #                 in_content = f.read().strip()
    #         except Exception:
    #             in_content = "[Could not read file]"
    #     else:
    #         in_content = "[File does not exist]"

    #     return (
    #         f"{boxed('mrgddb.in', in_content)}\n"
    #     )
