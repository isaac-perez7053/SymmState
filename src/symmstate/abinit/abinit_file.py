from . import AbinitUnitCell
import numpy as np
import os
import re
import subprocess
import copy
from symmstate.pseudopotentials.pseudopotential_manager import PseudopotentialManager
from symmstate import SymmStateCore
from typing import Optional, List
from symmstate.slurm_file import SlurmFile
import logging

class AbinitFile(AbinitUnitCell):
    """
    Class dedicated to writing and executing Abinit files.
    
    Revised functionality:
      - The user supplies a SlurmFile object (slurm_obj) which controls job submission,
        batch script creation, and holds running job IDs.
      - All messages are routed to the global logger.
      - Type hints and explicit type casting are used throughout.
    """

    def __init__(
        self,
        abi_file: Optional[str] = None,
        slurm_obj: Optional[SlurmFile] = None,
        convergence_file: Optional[str] = None,
        smodes_input: Optional[str] = None,
        target_irrep: Optional[str] = None,
        symmetry_informed_basis: Optional[bool] = False,
    ) -> None:
        # Initialize AbinitUnitCell with supported parameters.
        AbinitUnitCell.__init__(
            self,
            abi_file=abi_file,
            smodes_input=smodes_input,
            target_irrep=target_irrep,
        )
        self.convergence_file: Optional[str] = convergence_file
        self.symmetry_informed_basis: bool = symmetry_informed_basis

        if abi_file is not None:
            self.log_or_print(f"Name of abinit file: {abi_file}", logger=self._logger)
            self.file_name: str = str(abi_file).replace(".abi", "")
        else:
            self.file_name = "default_abinit_file"

        # Ensure atomic information is available from the structure.
        if hasattr(self, "structure"):
            self.natom: int = len(self.structure)
            unique_species: List = []
            for sp in self.structure.species:
                if sp not in unique_species:
                    unique_species.append(sp)
            self.ntypat: int = len(unique_species)
            self.znucl: List = [sp.Z for sp in self.structure.species]
            unique_z = sorted(set(self.znucl))
            self.typat: List = [unique_z.index(z) + 1 for z in self.znucl]

        # Instead of creating a SLURM script object internally, the user now supplies a SlurmFile instance.
        # Save it as self.slurm_obj. If none is supplied, log a warning.
        if slurm_obj is None:
            self.log_or_print("No SlurmFile object supplied; job submission may not work as intended.",
                              logger=self._logger, level=logging.WARNING)
        self.slurm_obj: Optional[SlurmFile] = slurm_obj

    @staticmethod
    def _get_unique_filename(file_name: str) -> str:
        """Generate a unique filename by appending a counter if the file exists."""
        base, ext = os.path.splitext(file_name)
        counter = 1
        unique_name = file_name
        while os.path.exists(unique_name):
            unique_name = f"{base}_{counter}{ext}"
            counter += 1
        return unique_name

    def write_custom_abifile(self, output_file: str, content: str, coords_are_cartesian: bool = False) -> None:
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
            outf.write(header_content)
            outf.write("\n#--------------------------\n# Definition of unit cell\n#--------------------------\n")
            acell = self.vars.get("acell", self.structure.lattice.abc) if hasattr(self, "vars") else self.structure.lattice.abc
            outf.write(f"acell {' '.join(map(str, acell))}\n")
            rprim = self.vars.get("rprim", self.structure.lattice.matrix.tolist()) if hasattr(self, "vars") else self.structure.lattice.matrix.tolist()
            outf.write("rprim\n")
            for coord in rprim:
                outf.write(f"  {'  '.join(map(str, coord))}\n")
            if coords_are_cartesian:
                outf.write("xcart\n")
                coordinates = self.grab_cartesian_coordinates()
                self.log_or_print(f"Coordinates to be written: {coordinates}", logger=self._logger)
                for coord in coordinates:
                    outf.write(f"  {'  '.join(map(str, coord))}\n")
            else:
                outf.write("xred\n")
                coordinates = self.grab_reduced_coordinates()
                for coord in coordinates:
                    outf.write(f"  {'  '.join(map(str, coord))}\n")
            outf.write("\n#--------------------------\n# Definition of atoms\n#--------------------------\n")
            outf.write(f"natom {self.natom} \n")
            outf.write(f"ntypat {self.ntypat} \n")
            outf.write(f"znucl {' '.join(map(str, self.znucl))}\n")
            outf.write(f"typat {' '.join(map(str, self.typat))}\n")
            if self.convergence_file is None:
                outf.write("\n#----------------------------------------\n# Definition of the planewave basis set\n#----------------------------------------\n")
                outf.write(f"ecut {self.ecut} \n")
                if hasattr(self, "ecutsm") and self.ecutsm is not None:
                    outf.write(f"ecutsm {self.vars['ecutsm']} \n")
                outf.write("\n#--------------------------\n# Definition of the k-point grid\n#--------------------------\n")
                outf.write(f"nshiftk {self.nshiftk} \n")
                outf.write("kptrlatt\n")
                if hasattr(self, "kptrlatt") and self.kptrlatt is not None:
                    for i in self.kptrlatt:
                        outf.write(f"  {' '.join(map(str, i))}\n")
                outf.write(f"shiftk {' '.join(map(str, self.shiftk))} \n")
                outf.write(f"nband {self.nband} \n")
                outf.write("\n#--------------------------\n# Definition of the SCF Procedure\n#--------------------------\n")
                outf.write(f"nstep {self.nstep} \n")
                outf.write(f"diemac {self.diemac} \n")
                outf.write(f"ixc {self.ixc} \n")
                outf.write(f"{self.conv_criteria[0]} {str(self.conv_criteria[1])} \n")
                # Use pseudopotential information parsed into self.vars.
                pp_dir = self.vars.get("pp_dirpath", "undefined")
                outf.write(f'\npp_dirpath "{pp_dir}" \n')
                pseudos = self.vars.get("pseudos", [])
                concatenated_pseudos = " ".join(pseudos)
                outf.write(f'pseudos "{concatenated_pseudos}" \n')
                self.log_or_print(f"The Abinit file {output_file} was created successfully!", logger=self._logger)
            else:
                with open(self.convergence_file, "r") as cf:
                    convergence_content = cf.read()
                outf.write(convergence_content)


    def run_abinit(
        self,
        input_file: str = "abinit",
        batch_name: str = "abinit_job",
        host_spec: str = "mpirun -hosts=localhost -np 30",
        delete_batch_script: bool = True,
        log: str = "log",
    ) -> None:
        """
        Executes the Abinit program using a generated input file and specified settings.
        """
        content: str = f"""{input_file}.abi
{input_file}.abo
{input_file}o
{input_file}_gen_output
{input_file}_temp
        """
        # We now require a SlurmFile object (self.slurm_obj) to handle batch script operations.
        if self.slurm_obj is not None:
            file_path: str = f"{input_file}_abinit_input_data.txt"
            file_path = AbinitFile._get_unique_filename(file_path)
            with open(file_path, "w") as file:
                file.write(content)
            try:
                batch_name = AbinitFile._get_unique_filename(f"{batch_name}.sh")
                batch_name = os.path.basename(batch_name)
                # Use the provided SlurmFile object.
                script_created = self.slurm_obj.write_batch_script(
                    input_file=file_path,
                    batch_name=batch_name,
                    host_spec=host_spec,
                    log=log,
                )
                self.log_or_print(f"Batch script created: {script_created}", logger=self._logger)
                result = subprocess.run(
                    ["sbatch", batch_name], capture_output=True, text=True
                )
                if result.returncode == 0:
                    self.log_or_print("Batch job submitted using 'sbatch'.", logger=self._logger)
                    try:
                        job_number = int(result.stdout.strip().split()[-1])
                        self.slurm_obj.running_jobs.append(job_number)
                        self.log_or_print(f"Job number {job_number} added to running jobs.", logger=self._logger)
                    except (ValueError, IndexError) as e:
                        self.log_or_print(f"Failed to parse job number: {e}", logger=self._logger, level=logging.ERROR)
                else:
                    self.log_or_print(f"Failed to submit batch job: {result.stderr}", logger=self._logger, level=logging.ERROR)
            finally:
                if delete_batch_script:
                    batch_script_path = f"{batch_name}.sh"
                    if os.path.exists(batch_script_path):
                        os.remove(batch_script_path)
                        self.log_or_print(f"Batch script '{batch_script_path}' has been removed.", logger=self._logger)
        else:
            # If no SlurmFile object was provided, execute directly.
            command: str = f"{host_spec} abinit < {input_file} > {log}"
            os.system(command)
            self.log_or_print(f"Abinit executed directly. Output written to '{log}'.", logger=self._logger)

    def run_piezo_calculation(self, host_spec: str = "mpirun -hosts=localhost -np 30") -> None:
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
        working_directory: str = os.getcwd()
        output_file: str = os.path.join(working_directory, f"{self.file_name}_piezo")
        batch_name: str = os.path.join(working_directory, f"{self.file_name}_bscript")
        self.write_custom_abifile(output_file=output_file, content=content, coords_are_cartesian=False)
        self.run_abinit(
            input_file=output_file,
            batch_name=batch_name,
            host_spec=host_spec,
            log="log"
        )

    def run_flexo_calculation(self, host_spec: str = "mpirun -hosts=localhost -np 30") -> None:
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
        working_directory: str = os.getcwd()
        output_file: str = os.path.join(working_directory, f"{self.file_name}_flexo")
        batch_name: str = os.path.join(working_directory, f"{self.file_name}_bscript")
        self.write_custom_abifile(output_file=output_file, content=content, coords_are_cartesian=False)
        self.run_abinit(
            input_file=output_file,
            batch_name=batch_name,
            host_spec=host_spec,
            log="log"
        )

    def run_energy_calculation(self, host_spec: str = "mpirun -hosts=localhost -np 20") -> None:
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
        working_directory: str = os.getcwd()
        output_file: str = os.path.join(working_directory, f"{self.file_name}_energy")
        batch_name: str = os.path.join(working_directory, f"{self.file_name}_bscript")
        self.write_custom_abifile(output_file=output_file, content=content, coords_are_cartesian=True)
        self.run_abinit(
            input_file=output_file,
            batch_name=batch_name,
            host_spec=host_spec,
            log=f"{output_file}.log"
        )

    def run_anaddb_file(self, ddb_file: str, content: str = "", flexo: bool = False, peizo: bool = False) -> str:
        """
        Executes an anaddb calculation, customized for flexoelectric or piezoelectric responses.
        """
        if flexo:
            content_f: str = """
! anaddb calculation of flexoelectric tensor
flexoflag 1
"""
            files_content: str = f"""{self.file_name}_flexo_anaddb.abi
{self.file_name}_flexo_output
{ddb_file}
dummy1
dummy2
dummy3
dummy4
"""
            with open(f"{self.file_name}_flexo_anaddb.abi", "w") as outf:
                outf.write(content_f)
            with open(f"{self.file_name}_flexo_anaddb.files", "w") as outff:
                outff.write(files_content)
            command: str = f"anaddb < {self.file_name}_flexo_anaddb.files > {self.file_name}_flexo_anaddb.log"
            try:
                subprocess.run(command, shell=True, check=True)
                self.log_or_print(f"Command executed successfully: {command}", logger=self._logger)
            except subprocess.CalledProcessError as e:
                self.log_or_print(f"An error occurred while executing the command: {e}", logger=self._logger, level=logging.ERROR)
            return f"{self.file_name}_flexo_output"
        elif peizo:
            content_p: str = """
! Input file for the anaddb code
elaflag 3
piezoflag 3
instrflag 1
"""
            files_content: str = f"""
{self.file_name}_piezo_anaddb.abi
{self.file_name}_piezo_output
{ddb_file}
dummy1
dummy2
dummy3
"""
            with open(f"{self.file_name}_piezo_anaddb.abi", "w") as outf:
                outf.write(content_p)
            with open(f"{self.file_name}_piezo_anaddb.files", "w") as outff:
                outff.write(files_content)
            command: str = f"anaddb < {self.file_name}_piezo_anaddb.files > {self.file_name}_piezo_anaddb.log"
            try:
                subprocess.run(command, shell=True, check=True)
                self.log_or_print(f"Command executed successfully: {command}", logger=self._logger)
            except subprocess.CalledProcessError as e:
                self.log_or_print(f"An error occurred while executing the command: {e}", logger=self._logger, level=logging.ERROR)
            return f"{self.file_name}_piezo_output"
        else:
            with open(f"{self.file_name}_anaddb.abi", "w") as outf:
                outf.write(content)
            command: str = f"anaddb < {self.file_name}_anaddb.files > {self.file_name}_anaddb.log"
            try:
                subprocess.run(command, shell=True, check=True)
                self.log_or_print(f"Command executed successfully: {command}", logger=self._logger)
            except subprocess.CalledProcessError as e:
                self.log_or_print(f"An error occurred while executing the command: {e}", logger=self._logger, level=logging.ERROR)

    def run_mrgddb_file(self, content: str) -> None:
        """
        Executes the mrgddb command using a provided input file content.
        """
        with open(f"{self.file_name}_mrgddb.in", "w") as outf:
            outf.write(content)
        command: str = f"mrgddb < {self.file_name}_mrgddb.in"
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.log_or_print(f"Command executed successfully: {command}", logger=self._logger)
            self.log_or_print(f"Output: {result.stdout.decode()}", logger=self._logger)
        except subprocess.CalledProcessError as e:
            self.log_or_print(f"An error occurred while executing the command: {e}", logger=self._logger, level=logging.ERROR)
            self.log_or_print(f"Error output: {e.stderr.decode()}", logger=self._logger, level=logging.ERROR)

    def grab_energy(self, abo_file: str) -> None:
        """
        Retrieves the total energy from a specified Abinit output file.
        """
        if abo_file is None:
            raise Exception("Please specify the abo file you are attempting to access")
        total_energy_value: Optional[str] = None
        try:
            with open(abo_file) as f:
                abo_content: str = f.read()
            match = re.search(r"total_energy\s*:\s*(-?\d+\.\d+E?[+-]?\d*)", abo_content)
            if match:
                total_energy_value = match.group(1)
                self.energy: float = float(total_energy_value)
            else:
                self.log_or_print("Total energy not found.", logger=self._logger, level=logging.WARNING)
        except FileNotFoundError:
            self.log_or_print(f"The file {abo_file} was not found.", logger=self._logger, level=logging.ERROR)

    def grab_flexo_tensor(self, anaddb_file: Optional[str] = None) -> None:
        """
        Retrieves the TOTAL flexoelectric tensor from the specified file.
        """
        if anaddb_file is None:
            anaddb_file = "file_name_energy.abo"
        flexo_tensor: Optional[np.ndarray] = None
        try:
            with open(anaddb_file) as f:
                abo_content: str = f.read()
            flexo_match = re.search(
                r"TOTAL flexoelectric tensor \(units= nC/m\)\s*\n\s+xx\s+yy\s+zz\s+yz\s+xz\s+xy\n((?:.*\n){9})",
                abo_content,
            )
            if flexo_match:
                tensor_strings = flexo_match.group(1).strip().split("\n")
                flexo_tensor = np.array([list(map(float, line.split()[1:])) for line in tensor_strings])
        except FileNotFoundError:
            self.log_or_print(f"The file {anaddb_file} was not found.", logger=self._logger, level=logging.ERROR)
        self.flexo_tensor = flexo_tensor

    def parse_tensor(self, tensor_str: str) -> np.ndarray:
        """
        Parses a tensor string into a NumPy array.
        """
        lines = tensor_str.strip().splitlines()
        tensor_data = []
        for line in lines:
            elements = line.split()
            if all(part.lstrip('-').replace('.', '', 1).isdigit() for part in elements):
                try:
                    numbers = [float(value) for value in elements]
                    tensor_data.append(numbers)
                except ValueError as e:
                    self.log_or_print(f"Could not convert line to numbers: {line}, Error: {e}", logger=self._logger, level=logging.ERROR)
                    raise
        return np.array(tensor_data)

    def grab_piezo_tensor(self, anaddb_file: Optional[str] = None) -> None:
        """
        Retrieves the clamped and relaxed ion piezoelectric tensors.
        """
        if anaddb_file is None:
            anaddb_file = f"{self.file_name}_energy.abo"
        piezo_tensor_clamped: Optional[np.ndarray] = None
        piezo_tensor_relaxed: Optional[np.ndarray] = None
        try:
            with open(anaddb_file) as f:
                abo_content: str = f.read()
            clamped_match = re.search(
                r"Proper piezoelectric constants \(clamped ion\) \(unit:c/m\^2\)\s*\n((?:\s*-?\d+\.\d+\s+\n?)+)",
                abo_content,
            )
            if clamped_match:
                clamped_strings = clamped_match.group(1).strip().split("\n")
                piezo_tensor_clamped = np.array([list(map(float, line.split())) for line in clamped_strings])
            relaxed_match = re.search(
                r"Proper piezoelectric constants \(relaxed ion\) \(unit:c/m\^2\)\s*\n((?:\s*-?\d+\.\d+\s+\n?)+)",
                abo_content,
            )
            if relaxed_match:
                relaxed_strings = relaxed_match.group(1).strip().split("\n")
                piezo_tensor_relaxed = np.array([list(map(float, line.split())) for line in relaxed_strings])
        except FileNotFoundError:
            self.log_or_print(f"The file {anaddb_file} was not found.", logger=self._logger, level=logging.ERROR)
        self.piezo_tensor_clamped = piezo_tensor_clamped
        self.piezo_tensor_relaxed = piezo_tensor_relaxed

    def clean_files(self, filename: str = "filename.abi") -> None:
        pass

    def copy_abinit_file(self):
        """
        Creates a deep copy of the current AbinitFile instance.
        """
        copied_file = copy.deepcopy(self)
        return copied_file

