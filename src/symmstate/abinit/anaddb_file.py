import os
import subprocess

from symmstate.utils import get_unique_filename


class AnaddbFile:
    """
    Write input and files for the Anaddb utility and to execute Anaddb
    """

    def __init__(self):
        pass

    @staticmethod
    def write(
        in_content: str,
        anaddb_in: str,
        directory: str = ".",
        out_name: str = "anaddb.out",
        ddb: str = "ddb",
        band_eps: str = "band_eps",
        gkk: str = "gkk",
        ep: str = "anaddb.ep",
        ddk: str = "ddk",
        files_name: str = "anaddb.files",
    ) -> str:
        """
        Write both the anaddb.in and anaddb.files files.

        Parameters:
            in_content (str):
                Content to write to the anaddb.in file.
            anaddb_in (str):
                Name of the anaddb.in file.
            directory (Optional[str]):
                Directory in which to write the files, by default "."
            out_name (Optional[str]):
                Name of the output file, by default "anaddb.out"
            ddb (Optional[str]):
                Name of the ddb file, by default "ddb"
            band_eps (Optional[str]):
                Name of the band_eps file, by default "band_eps"
            gkk (Optional[str]):
                Name of the gkk file, by default "gkk"
            ep (Optional[str]):
                Name of the anaddb.ep file, by default "anaddb.ep"
            ddk (Optional[str]):
                Name of the ddk file, by default "ddk"
            files_name (Optional[str]):
                Name of the anaddb.files file, by default "anaddb.files"

        Returns:
            str: The unique name of the output file
        """
        out_name = get_unique_filename(out_name)

        # Compute all paths
        in_path = os.path.join(directory, anaddb_in)
        out_path = os.path.join(directory, out_name)
        ddb_path = os.path.join(directory, ddb)
        band_eps_path = os.path.join(directory, band_eps)
        gkk_path = os.path.join(directory, gkk)
        ep_path = os.path.join(directory, ep)
        ddk_path = os.path.join(directory, ddk)
        files_path = os.path.join(directory, files_name)

        # Write the anaddb.in file
        with open(in_path, "w") as f:
            f.write(in_content)

        # Write the anaddb.files file
        with open(files_path, "w") as f:
            f.write(
                "\n".join(
                    [
                        in_path,
                        out_path,
                        ddb_path,
                        band_eps_path,
                        gkk_path,
                        ep_path,
                        ddk_path,
                    ]
                )
                + "\n"
            )

        return out_name

    @staticmethod
    def execute(
        files_name: str = "anaddb.files",
        log_name: str = "anaddb.log",
        directory: str = ".",
        background: bool = False,
    ) -> None:
        """
        Execute the anaddb utility using the specified files and log.

        Parameters:
            files_name (str):
                Name of the anaddb.files file to read, by default "anaddb.files"
            log_name (Optional[str]):
                Name of the log file to write, by default "anaddb.log"
            directory (Optional[str]):
                Directory containing the files, by default "."
            background (Optional[bool]):
                If True, run the command in the background, by default False

        Returns:
            None
        """
        files_path = os.path.join(directory, files_name)
        log_path = os.path.join(directory, log_name)
        command = f"anaddb < {files_path} >& {log_path}"
        if background:
            command += " &"
        subprocess.run(command, shell=True, check=not background)

    # def __repr__(self):
    #     def boxed(title, content):
    #         lines = content.splitlines() if content else []
    #         width = max([len(title)] + [len(line) for line in lines]) + 4
    #         top = f"┌{'─' * (width - 2)}┐"
    #         mid = f"│ {title.ljust(width - 3)}│"
    #         body = "\n".join(f"│ {line.ljust(width - 3)}│" for line in lines) if lines else f"│ {'(empty)'.ljust(width - 3)}│"
    #         bottom = f"└{'─' * (width - 2)}┘"
    #         return f"{top}\n{mid}\n{body}\n{bottom}"

    #     # anaddb.in content
    #     if os.path.isfile(self.in_path):
    #         try:
    #             with open(self.in_path, "r") as f:
    #                 in_content = f.read().strip()
    #         except Exception:
    #             in_content = "[Could not read file]"
    #     else:
    #         in_content = "[File does not exist]"

    #     # anaddb.files content
    #     if os.path.isfile(self.files_path):
    #         try:
    #             with open(self.files_path, "r") as f:
    #                 files_content = f.read().strip()
    #         except Exception:
    #             files_content = "[Could not read file]"
    #     else:
    #         files_content = "[File does not exist]"

    #     return (
    #         f"{boxed('anaddb.in', in_content)}\n"
    #         f"{boxed('anaddb.files', files_content)}\n"
    #     )
