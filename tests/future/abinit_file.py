from . import AbinitUnitCell
import os
import subprocess
import copy
from symmstate.pseudopotentials.pseudopotential_manager import PseudopotentialManager
from typing import Optional, List
from symmstate.slurm import *
from pymatgen.core import Structure
from symmstate.config.symm_state_settings import settings


class AbinitFile(AbinitUnitCell):
    """
    Class dedicated to writing and executing Abinit files.

      - The user supplies a SlurmFile object (slurm_obj) which controls job submission,
        batch script creation, and holds running job IDs.
      - All messages are routed to the global logger.
      - Type hints and explicit type casting are used throughout.
    """

    @staticmethod
    def _get_unique_filename(base_name: str) -> str:
        # Example: just add a suffix or something more robust in real code
        return f"{base_name}_unique"

    def __init__(
        self,
        abi_file: Optional[str] = None,
        unit_cell: Optional[Structure] = None,
        *,
        smodes_input: Optional[str] = None,
        target_irrep: Optional[str] = None,
    ) -> None:
        # Initialize AbinitUnitCell with supported parameters.
        AbinitUnitCell.__init__(
            self,
            abi_file=abi_file,
            unit_cell=unit_cell,
            smodes_input=smodes_input,
            target_irrep=target_irrep,
        )

        if abi_file is not None:
            self._logger.info(f"Name of abinit file: {abi_file}")
            self.file_name: str = str(abi_file).replace(".abi", "")
        else:
            self.file_name = "abinit_file"

    def _abs_path(self, *subpaths: str) -> str:
        """
        Joins subpaths onto settings.PROJECT_ROOT to form an absolute path.
        Example usage: self._abs_path("data", "my_file.txt")
        """
        return os.path.join(settings.PROJECT_ROOT, *subpaths)

    def write_custom_abifile(
        self,
        output_file: str,
        content: str,
        coords_are_cartesian: bool = False,
        pseudos: List = [],
    ) -> None:
        """
        Writes a custom Abinit .abi file using user-defined or default parameters.

        Args:
            output_file (str): Path where the new Abinit file will be saved.
            content (str): Header content or path to a header file.
            coords_are_cartesian (bool): Flag indicating the coordinate system.
        """
        # Determine whether 'content' is literal text or a file path.
        if "\n" in content or not os.path.exists(content):
            header_content: str = content
        else:
            with open(content, "r") as hf:
                header_content = hf.read()

        # Generate a unique filename.
        output_file = AbinitFile._get_unique_filename(output_file)

        with open(f"{output_file}.abi", "w") as outf:
            # Write the header content
            outf.write(header_content)
            outf.write(
                "\n#--------------------------\n# Definition of unit cell\n#--------------------------\n"
            )
            acell = self.vars.get("acell", self.structure.lattice.abc)
            outf.write(f"acell {' '.join(map(str, acell))}\n")
            rprim = self.vars.get("rprim", self.structure.lattice.matrix.tolist())
            outf.write("rprim\n")

            for coord in rprim:
                outf.write(f"  {'  '.join(map(str, coord))}\n")

            if coords_are_cartesian:
                outf.write("xcart\n")
                coordinates = self.vars["xcart"]
                self._logger.info(f"Coordinates to be written: {coordinates}")

                for coord in coordinates:
                    outf.write(f"  {'  '.join(map(str, coord))}\n")

            else:
                outf.write("xred\n")
                coordinates = self.vars["xred"]
                for coord in coordinates:
                    outf.write(f"  {'  '.join(map(str, coord))}\n")

            outf.write(
                "\n#--------------------------\n# Definition of atoms\n#--------------------------\n"
            )
            outf.write(f"natom {self.vars['natom']} \n")
            outf.write(f"ntypat {self.vars['ntypat']} \n")
            outf.write(f"znucl {' '.join(map(str, self.vars['znucl']))}\n")
            outf.write(f"typat {' '.join(map(str, self.vars['typat']))}\n")

            outf.write(
                "\n#----------------------------------------\n# Definition of the planewave basis set\n#----------------------------------------\n"
            )
            outf.write(f"ecut {self.vars.get('ecut', 42)} \n")
            if self.vars["ecutsm"] is not None:
                outf.write(f"ecutsm {self.vars['ecutsm']} \n")

            outf.write(
                "\n#--------------------------\n# Definition of the k-point grid\n#--------------------------\n"
            )
            outf.write(f"nshiftk {self.vars.get('nshiftk', '1')} \n")
            outf.write("kptrlatt\n")
            if self.vars["kptrlatt"] is not None:
                for i in self.vars["kptrlatt"]:
                    outf.write(f"  {' '.join(map(str, i))}\n")
            outf.write(
                f"shiftk {' '.join(map(str, self.vars.get('shiftk', '0.5 0.5 0.5')))} \n"
            )
            outf.write(f"nband {self.vars['nband']} \n")

            outf.write(
                "\n#--------------------------\n# Definition of the SCF Procedure\n#--------------------------\n"
            )
            outf.write(f"nstep {self.vars.get('nstep', 9)} \n")
            outf.write(f"diemac {self.vars.get('diemac', '1000000.0')} \n")
            outf.write(f"ixc {self.vars['ixc']} \n")
            outf.write(
                f"{self.vars['conv_criteria']} {str(self.vars[self.vars['conv_criteria']])} \n"
            )
            # Use pseudopotential information parsed into self.vars.
            pp_dir_path = PseudopotentialManager().folder_path
            outf.write(f'\npp_dirpath "{pp_dir_path}" \n')
            if len(pseudos) == 0:
                pseudos = self.vars.get("pseudos", [])
            concatenated_pseudos = ", ".join(pseudos).replace('"', "")
            outf.write(f'pseudos "{concatenated_pseudos}"\n')
            self._logger.info(
                f"The Abinit file {output_file} was created successfully!"
            )

    ############################################################################
    # 1) run_abinit
    ############################################################################
    def run_abinit(
        self,
        input_file: str,
        slurm_obj: Optional[SlurmFile] = None,
        *,
        batch_name: Optional[str],
        log_file: Optional[str],
        extra_commands: Optional[str] = None,
    ) -> None:
        """
        Executes the Abinit program using a generated input file and specified settings.
        """
        # Just as an example, let's create some arbitrary content referencing the input file
        content: str = f"""{input_file}.abi
{input_file}.abo
{input_file}o
{input_file}_gen_output
{input_file}_temp
        """

        # Convert input_file, batch_name, log_file to absolute paths for consistency
        # Typically you'd store data or scripts in subdirectories like "data/", "scripts/", "logs/", etc.
        input_file_abs = self._abs_path("data", input_file)
        batch_name_abs = None
        log_file_abs = None

        if batch_name:
            batch_name_abs = self._abs_path("scripts", f"{batch_name}.sh")
        if log_file:
            log_file_abs = self._abs_path("logs", log_file)

        # If using SlurmFile (i.e., HPC run)
        if slurm_obj is not None:
            # Let's create a unique text file in "temp/" for your content
            file_path = f"{input_file}_abinit_input_data.txt"
            file_path = AbinitFile._get_unique_filename(file_path)
            file_path_abs = self._abs_path("temp", file_path)

            # Write out 'content' to that file
            with open(file_path_abs, "w") as file:
                file.write(content)

            try:
                # Make the batch script name unique as well
                if batch_name_abs:
                    unique_batch = AbinitFile._get_unique_filename(
                        os.path.basename(batch_name_abs)
                    )
                    batch_name_abs = self._abs_path("scripts", unique_batch)

                # Actually write the batch script with SlurmFile
                script_created = slurm_obj.write_batch_script(
                    input_file=f"{input_file_abs}.abi",  # Possibly your real .abi file
                    log_file=log_file_abs if log_file_abs else "/dev/null",
                    batch_name=(
                        batch_name_abs
                        if batch_name_abs
                        else self._abs_path("scripts", "default_batch.sh")
                    ),
                    extra_commands=extra_commands,
                )
                self._logger.info(f"Batch script created: {script_created}")

                # Submit the SLURM job
                slurm_obj.submit_job(script_created)
            finally:
                # For demonstration, we keep your original logic that forcibly raises
                raise RuntimeError("Failed to run abinit using the batch script!")

        else:
            # If no SlurmFile is provided, execute directly with abinit, writing output to log_file
            log_file_final = log_file_abs if log_file_abs else "abinit.log"
            command: str = f"abinit {input_file_abs} > {log_file_final}"
            os.system(command)
            self._logger.info(
                f"Abinit executed directly. Output written to '{log_file_final}'."
            )

    ############################################################################
    # 2) run_piezo_calculation
    ############################################################################
    def run_piezo_calculation(
        self,
        slurm_obj: Optional[SlurmFile] = None,
        *,
        batch_name: Optional[str] = None,
        log_file: Optional[str] = None,
        extra_commands: Optional[str] = None,
    ) -> None:
        """
        Runs a piezoelectricity calculation for the unit cell.
        """
        content: str = """ndtset 2
chkprim 0

# Set 1 : Ground State Self-Consistent Calculation
#************************************************
  kptopt1 1
  tolvrs 1.0d-18

# Set 2 : Calculation of ddk wavefunctions
#************************************************
  kptopt2 2
  getwfk2 1
  rfelfd2 2
  iscf2   -3
  tolwfr2 1.0D-18
"""
        # Store the output file in "data/"
        output_file_abs = self._abs_path("data", f"{self.file_name}_piezo")

        # Let batch_name default to something if not set
        batch_name = batch_name or f"{self.file_name}_bscript"

        # Example: store script in "scripts/"
        batch_name_abs = self._abs_path("scripts", batch_name)

        # Write the custom .abi file
        self.write_custom_abifile(
            output_file=output_file_abs, content=content, coords_are_cartesian=False
        )

        # Now call run_abinit
        self.run_abinit(
            input_file=output_file_abs,
            slurm_obj=slurm_obj,
            batch_name=batch_name_abs,
            log_file=log_file or "piezo.log",  # store in logs/ eventually
            extra_commands=extra_commands,
        )

    ############################################################################
    # 3) run_flexo_calculation
    ############################################################################
    def run_flexo_calculation(
        self, host_spec: str = "mpirun -hosts=localhost -np 30"
    ) -> None:
        """
        Runs a flexoelectricity calculation for the unit cell.
        """
        content: str = """ndtset 5
chkprim 0

# Set 1: Ground State Self-Consistency
#*************************************
getwfk1 0
kptopt1 1
tolvrs1 1.0d-18

# Set 2: Response function calculation of d/dk wave function
#**********************************************************
iscf2 -3
rfelfd2 2
tolwfr2 1.0d-20

# Set 3: Response function calculation of d2/dkdk wavefunction
#*************************************************************
getddk3 2
iscf3 -3
rf2_dkdk3 3
tolwfr3 1.0d-16
rf2_pert1_dir3 1 1 1
rf2_pert2_dir3 1 1 1

# Set 4: Response function calculation to q=0 phonons, electric field and strain
#*******************************************************************************
getddk4 2
rfelfd4 3
rfphon4 1
rfstrs4 3
rfstrs_ref4 1
tolvrs4 1.0d-8
prepalw4 1

getwfk 1
useylm 1
kptopt2 2
"""
        output_file_abs = self._abs_path("data", f"{self.file_name}_flexo")
        batch_name_abs = self._abs_path("scripts", f"{self.file_name}_bscript")

        # Write the custom .abi file
        self.write_custom_abifile(
            output_file=output_file_abs, content=content, coords_are_cartesian=False
        )

        # Now run abinit
        # Note: host_spec is presumably the MPI command template or something similar
        self.run_abinit(
            input_file=output_file_abs,
            batch_name=batch_name_abs,
            host_spec=host_spec,  # might need to adjust your function signature to accept this
            log_file="flexo.log",
        )

    ############################################################################
    # 4) run_energy_calculation
    ############################################################################
    def run_energy_calculation(self) -> None:
        """
        Runs an energy calculation for the unit cell.
        """
        content: str = """ndtset 1
chkprim 0

# Ground State Self-Consistency
#*******************************
getwfk1 0
kptopt1 1

# Turn off various file outputs
prtpot 0
prteig 0

getwfk 1
useylm 1
kptopt2 2
"""
        output_file_abs = self._abs_path("data", f"{self.file_name}_energy")
        batch_name_abs = self._abs_path("scripts", f"{self.file_name}_bscript")
        log_file_abs = f"{output_file_abs}.log"

        self.write_custom_abifile(
            output_file=output_file_abs, content=content, coords_are_cartesian=True
        )

        # Provide 'host_spec' via self.slurm_obj.mpi_command_template or however you handle it
        # If you want to run HPC or local, ensure self.slurm_obj is set or pass it in
        self.run_abinit(
            input_file=output_file_abs,
            batch_name=batch_name_abs,
            host_spec=self.slurm_obj.mpi_command_template if self.slurm_obj else None,
            log_file=log_file_abs,
        )

    ############################################################################
    # 5) run_anaddb_file
    ############################################################################
    def run_anaddb_file(
        self,
        content: str = "",
        files_content: str = "",
        *,
        ddb_file: str,
        flexo: bool = False,
        peizo: bool = False,
    ) -> str:
        """
        Executes an anaddb calculation. Supports default manual mode and optional presets
        for flexoelectric or piezoelectric calculations.

        Args:
            ddb_file: Path to the DDB file.
            content: Content to write into the .abi file (used if neither flexo nor peizo are True).
            files_content: Content for the .files file (used if neither flexo nor peizo are True).
            flexo: If True, runs a flexoelectric preset calculation.
            peizo: If True, runs a piezoelectric preset calculation.

        Returns:
            str: Name of the output file produced.
        """
        # Adjust to absolute path for ddb_file if needed
        ddb_file_abs = self._abs_path("data", ddb_file)

        if flexo:
            content = """
! anaddb calculation of flexoelectric tensor
flexoflag 1
""".strip()

            files_content = f"""{self.file_name}_flexo_anaddb.abi
{self.file_name}_flexo_output
{ddb_file_abs}
dummy1
dummy2
dummy3
dummy4
""".strip()

            abi_path = f"{self.file_name}_flexo_anaddb.abi"
            files_path = f"{self.file_name}_flexo_anaddb.files"
            log_path = f"{self.file_name}_flexo_anaddb.log"
            output_file = f"{self.file_name}_flexo_output"

        elif peizo:
            content = """
! Input file for the anaddb code
elaflag 3
piezoflag 3
instrflag 1
""".strip()

            files_content = f"""{self.file_name}_piezo_anaddb.abi
{self.file_name}_piezo_output
{ddb_file_abs}
dummy1
dummy2
dummy3
""".strip()

            abi_path = f"{self.file_name}_piezo_anaddb.abi"
            files_path = f"{self.file_name}_piezo_anaddb.files"
            log_path = f"{self.file_name}_piezo_anaddb.log"
            output_file = f"{self.file_name}_piezo_output"

        else:
            if not content.strip() or not files_content.strip():
                raise ValueError(
                    "Must provide both `content` and `files_content` when not using flexo or peizo mode."
                )

            abi_path = f"{self.file_name}_anaddb.abi"
            files_path = f"{self.file_name}_anaddb.files"
            log_path = f"{self.file_name}_anaddb.log"
            output_file = f"{self.file_name}_anaddb_output"

        # Convert them to absolute paths
        abi_path_abs = self._abs_path("data", abi_path)
        files_path_abs = self._abs_path("data", files_path)
        log_path_abs = self._abs_path("logs", log_path)

        # Write the .abi and .files
        with open(abi_path_abs, "w") as abi_file:
            abi_file.write(content)
        with open(files_path_abs, "w") as files_file:
            files_file.write(files_content)

        # anaddb command with absolute references
        command = f"anaddb < {files_path_abs} > {log_path_abs}"
        try:
            subprocess.run(command, shell=True, check=True)
            self._logger.info(f"Command executed successfully: {command}")
        except subprocess.CalledProcessError as e:
            self._logger.error(f"An error occurred while executing the command: {e}")

        return output_file

    ############################################################################
    # copy_abinit_file
    ############################################################################
    def copy_abinit_file(self):
        """
        Creates a deep copy of the current AbinitFile instance.

        Returns:
            AbinitFile: A new instance that is a deep copy of self.
        """
        # Example placeholder if you want to duplicate the object
        import copy

        return copy.deepcopy(self)

    def __repr__(self):

        lines = []
        lines.append("#--------------------------")
        lines.append("# Definition of unit cell")
        lines.append("#--------------------------")
        acell = self.vars.get("acell", self.structure.lattice.abc)
        lines.append(f"acell {' '.join(map(str, acell))}")
        rprim = self.vars.get("rprim", self.structure.lattice.matrix.tolist())
        lines.append("rprim")
        for coord in rprim:
            lines.append(f"  {'  '.join(map(str, coord))}")
        # Choose coordinate system: xcart if available; otherwise xred.
        if self.vars.get("xcart") is not None:
            lines.append("xcart")
            coordinates = self.vars["xcart"]
            for coord in coordinates:
                lines.append(f"  {'  '.join(map(str, coord))}")
        else:
            lines.append("xred")
            coordinates = self.vars.get("xred", [])
            for coord in coordinates:
                lines.append(f"  {'  '.join(map(str, coord))}")
        lines.append("")
        lines.append("#--------------------------")
        lines.append("# Definition of atoms")
        lines.append("#--------------------------")
        lines.append(f"natom {self.vars.get('natom')}")
        lines.append(f"ntypat {self.vars.get('ntypat')}")
        lines.append(f"znucl {' '.join(map(str, self.vars.get('znucl', [])))}")
        lines.append(f"typat {' '.join(map(str, self.vars.get('typat', [])))}")
        lines.append("")
        lines.append("#----------------------------------------")
        lines.append("# Definition of the planewave basis set")
        lines.append("#----------------------------------------")
        lines.append(f"ecut {self.vars.get('ecut', 42)}")
        if self.vars.get("ecutsm") is not None:
            lines.append(f"ecutsm {self.vars.get('ecutsm')}")
        lines.append("")
        lines.append("#--------------------------")
        lines.append("# Definition of the k-point grid")
        lines.append("#--------------------------")
        lines.append(f"nshiftk {self.vars.get('nshiftk', '1')}")
        lines.append("kptrlatt")
        if self.vars.get("kptrlatt") is not None:
            for row in self.vars.get("kptrlatt"):
                lines.append(f"  {' '.join(map(str, row))}")
        # Make sure to split shiftk if it's a string
        shiftk = self.vars.get("shiftk", "0.5 0.5 0.5")
        if isinstance(shiftk, str):
            shiftk = shiftk.split()
        lines.append(f"shiftk {' '.join(map(str, shiftk))}")
        lines.append(f"nband {self.vars.get('nband')}")
        lines.append("")
        lines.append("#--------------------------")
        lines.append("# Definition of the SCF Procedure")
        lines.append("#--------------------------")
        lines.append(f"nstep {self.vars.get('nstep', 9)}")
        lines.append(f"diemac {self.vars.get('diemac', '1000000.0')}")
        lines.append(f"ixc {self.vars.get('ixc')}")
        conv_criteria = self.vars.get("conv_criteria")
        if conv_criteria is not None:
            conv_value = self.vars.get(conv_criteria)
            lines.append(f"{conv_criteria} {str(conv_value)}")
        pp_dir_path = PseudopotentialManager().folder_path
        lines.append(f'pp_dirpath "{pp_dir_path}"')
        pseudos = self.vars.get("pseudos", [])
        # Remove any embedded double quotes from each pseudo and then join them.
        concatenated_pseudos = ", ".join(pseudo.replace('"', "") for pseudo in pseudos)
        lines.append(f'pseudos "{concatenated_pseudos}"')
        return "\n".join(lines)
