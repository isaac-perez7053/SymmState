from . import AbinitUnitCell
import numpy as np
import os
import re
import subprocess
import copy
from symmstate.slurm_file import SlurmFile
from symmstate.pseudopotentials.pseudopotential_manager import PseudopotentialManager

class AbinitFile(AbinitUnitCell, SlurmFile):
    """
    Class dedicated to writing and executing Abinit files.
    This version reflects the revised functionality of the dependencies:
      - AbinitUnitCell (updated initialization, parsing, and unit cell handling)
      - SlurmFile (updated SLURM job management)
      - UnitCell (from unit_cell_module) and TemplateManager (for templating)
    """

    def __init__(
        self,
        abi_file=None,
        batch_script_header_file=None,
        convergence_file=None,
        smodes_input=None,
        target_irrep=None,
        symmetry_informed_basis=False,
    ):
        # Initialize AbinitUnitCell with supported parameters (drop unsupported ones)
        AbinitUnitCell.__init__(
            self,
            abi_file=abi_file,
            smodes_input=smodes_input,
            target_irrep=target_irrep,
        )
        # Store additional parameters specific to AbinitFile
        self.convergence_file = convergence_file
        self.symmetry_informed_basis = symmetry_informed_basis

        if abi_file is not None:
            print(f"Name of abinit file: {abi_file}")
            self.file_name = abi_file.replace(".abi", "")

        # Ensure atomic information is available from the structure
        if hasattr(self, "structure"):
            self.natom = len(self.structure)
            # Build a list of unique species for ntypat determination
            unique_species = []
            for sp in self.structure.species:
                if sp not in unique_species:
                    unique_species.append(sp)
            self.ntypat = len(unique_species)
            # Set znucl as the atomic numbers for each atom
            self.znucl = [sp.Z for sp in self.structure.species]
            # typat: index of each atom's atomic number in the sorted unique list (starting at 1)
            unique_z = sorted(set(self.znucl))
            self.typat = [unique_z.index(z) + 1 for z in self.znucl]

        # Initialize the SLURM batch file functionality
        SlurmFile.__init__(self, batch_script_header_file)

    @staticmethod
    def _get_unique_filename(file_name):
        """Generate a unique filename by appending a counter if the file exists."""
        base, ext = os.path.splitext(file_name)
        counter = 1
        unique_name = file_name
        while os.path.exists(unique_name):
            unique_name = f"{base}_{counter}{ext}"
            counter += 1
        return unique_name

    def write_custom_abifile(self, output_file, content, coords_are_cartesian=False):
        """
        Writes a custom Abinit .abi file using user-defined or default parameters.
        Depending on whether convergence_file is provided, either standard input sections
        or the convergence content will be written.

        Args:
            output_file (str): Path where the new Abinit file will be saved.
            content (str): Header content or path to a header file.
            coords_are_cartesian (bool): Flag to determine coordinate system.
        """
        # Determine if content is actual content or a file path
        if "\n" in content or not os.path.exists(content):
            header_content = content
        else:
            with open(content, "r") as hf:
                header_content = hf.read()

        # Get a unique filename to avoid overwrites
        output_file = AbinitFile._get_unique_filename(output_file)

        with open(f"{output_file}.abi", "w") as outf:
            outf.write(header_content)

            # Write unit cell definition using updated parameters
            outf.write("\n#--------------------------")
            outf.write("\n# Definition of unit cell")
            outf.write("\n#--------------------------\n")
            # Use the acell from parsed variables or lattice parameters
            acell = self.vars.get("acell", self.structure.lattice.abc) if hasattr(self, "vars") else self.structure.lattice.abc
            outf.write(f"acell {' '.join(map(str, acell))}\n")
            # Use the rprim from parsed variables or lattice matrix
            rprim = self.vars.get("rprim", self.structure.lattice.matrix.tolist()) if hasattr(self, "vars") else self.structure.lattice.matrix.tolist()
            outf.write("rprim\n")
            for coord in rprim:
                outf.write(f"  {'  '.join(map(str, coord))}\n")

            # Write coordinates (either cartesian or reduced)
            if coords_are_cartesian:
                outf.write("xcart\n")
                coordinates = self.grab_cartesian_coordinates()
                print("Coordinates to be written:", coordinates)
                for coord in coordinates:
                    outf.write(f"  {'  '.join(map(str, coord))}\n")
            else:
                outf.write("xred\n")
                coordinates = self.grab_reduced_coordinates()
                for coord in coordinates:
                    outf.write(f"  {'  '.join(map(str, coord))}\n")

            # Write atom information
            outf.write("\n#--------------------------")
            outf.write("\n# Definition of atoms")
            outf.write("\n#--------------------------\n")
            outf.write(f"natom {self.natom} \n")
            outf.write(f"ntypat {self.ntypat} \n")
            outf.write(f"znucl {' '.join(map(str, self.znucl))}\n")
            outf.write(f"typat {' '.join(map(str, self.typat))}\n")

            if self.convergence_file is None:
                outf.write("\n#----------------------------------------")
                outf.write("\n# Definition of the planewave basis set")
                outf.write("\n#----------------------------------------\n")
                outf.write(f"ecut {self.ecut} \n")
                if hasattr(self, "ecutsm") and self.ecutsm is not None:
                    outf.write(f"ecutsm {self.ecutsm} \n")

                outf.write("\n#--------------------------")
                outf.write("\n# Definition of the k-point grid")
                outf.write("\n#--------------------------\n")
                outf.write(f"nshiftk {self.nshiftk} \n")
                outf.write("kptrlatt\n")
                if hasattr(self, "kptrlatt") and self.kptrlatt is not None:
                    for i in self.kptrlatt:
                        outf.write(f"  {' '.join(map(str, i))}\n")
                outf.write(f"shiftk {' '.join(map(str, self.shiftk))} \n")
                outf.write(f"nband {self.nband} \n")
                outf.write("\n#--------------------------")
                outf.write("\n# Definition of the SCF Procedure")
                outf.write("\n#--------------------------\n")
                outf.write(f"nstep {self.nstep} \n")
                outf.write(f"diemac {self.diemac} \n")
                outf.write(f"ixc {self.ixc} \n")
                outf.write(f"{self.conv_criteria[0]} {str(self.conv_criteria[1])} \n")

                pp_manager = PseudopotentialManager()
                outf.write(f'\npp_dirpath "{pp_manager.folder_path}" \n')
                if hasattr(self, "pseudopotentials") and self.pseudopotentials:
                    pseudos = self.pseudopotentials
                else:
                    pseudos = list(pp_manager.pseudo_registry.keys())
                concatenated_pseudos = " ".join(pseudos)
                outf.write(f'pseudos "{concatenated_pseudos}" \n')
                print(f"The Abinit file {output_file} was created successfully! \n")

            else:
                with open(self.convergence_file, "r") as cf:
                    convergence_content = cf.read()
                outf.write(convergence_content)

    def run_abinit(
        self,
        input_file="abinit",
        batch_name="abinit_job",
        batch_script_header_file=None,
        host_spec="mpirun -hosts=localhost -np 30",
        delete_batch_script=True,
        log="log",
    ):
        """
        Executes the Abinit program using a generated input file and specified settings.
        """
        # [Job execution code remains unchanged]
        content = f"""{input_file}.abi
{input_file}.abo
{input_file}o
{input_file}_gen_output
{input_file}_temp
    """
        if batch_script_header_file is not None:
            file_path = f"{input_file}_abinit_input_data.txt"
            file_path = AbinitFile._get_unique_filename(file_path)
            with open(file_path, "w") as file:
                file.write(content)
            try:
                batch_name = AbinitFile._get_unique_filename(f"{batch_name}.sh")
                batch_name = os.path.basename(batch_name)
                script_created = self.write_batch_script(
                    input_file=file_path,
                    batch_name=batch_name,
                    host_spec=host_spec,
                    log=log,
                )
                print(f"Was the batch script successfully created: {script_created} \n")
                result = subprocess.run(
                    ["sbatch", batch_name], capture_output=True, text=True
                )
                if result.returncode == 0:
                    print("Batch job submitted using 'sbatch'.")
                    try:
                        job_number = int(result.stdout.strip().split()[-1])
                        self.running_jobs.append(job_number)
                        print(f"Job number {job_number} added to running jobs.")
                    except (ValueError, IndexError) as e:
                        print(f"Failed to parse job number: {e}")
                else:
                    print("Failed to submit batch job:", result.stderr)
            finally:
                if delete_batch_script:
                    batch_script_path = f"{batch_name}.sh"
                    if os.path.exists(batch_script_path):
                        os.remove(batch_script_path)
                        print(f"Batch script '{batch_script_path}' has been removed.")
        else:
            command = f"{host_spec} abinit < {input_file} > {log}"
            os.system(command)
            print(f"Abinit executed directly. Output written to '{log}'.")


    def run_piezo_calculation(self, host_spec="mpirun -hosts=localhost -np 30"):
        """
        Runs a piezoelectricity calculation for the unit cell.

        Args:
            host_spec (str): Parallel execution specification.
        """
        content = """ndtset 2
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
        working_directory = os.getcwd()
        output_file = os.path.join(working_directory, f"{self.file_name}_piezo")
        batch_name = os.path.join(working_directory, f"{self.file_name}_bscript")
        self.write_custom_abifile(output_file=output_file, content=content, coords_are_cartesian=False)
        self.run_abinit(
            input_file=output_file,
            batch_name=batch_name,
            host_spec=host_spec,
            batch_script_header_file=self.batch_header,
        )

    def run_flexo_calculation(self, host_spec="mpirun -hosts=localhost -np 30"):
        """
        Runs a flexoelectricity calculation for the unit cell.

        Args:
            host_spec (str): Parallel execution specification.
        """
        content = """ndtset 5
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
kptopt 2
"""
        working_directory = os.getcwd()
        output_file = os.path.join(working_directory, f"{self.file_name}_flexo")
        batch_name = os.path.join(working_directory, f"{self.file_name}_bscript")
        self.write_custom_abifile(output_file=output_file, content=content, coords_are_cartesian=False)
        self.run_abinit(
            input_file=output_file,
            batch_name=batch_name,
            host_spec=host_spec,
            batch_script_header_file=self.batch_header,
        )

    def run_energy_calculation(self, host_spec="mpirun -hosts=localhost -np 20"):
        """
        Runs an energy calculation for the unit cell.

        Args:
            host_spec (str): Parallel execution specification.
        """
        content = """ndtset 1
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
kptopt 2
"""
        working_directory = os.getcwd()
        output_file = os.path.join(working_directory, f"{self.file_name}_energy")
        batch_name = os.path.join(working_directory, f"{self.file_name}_bscript")
        self.write_custom_abifile(output_file=output_file, content=content, coords_are_cartesian=True)
        self.run_abinit(
            input_file=output_file,
            batch_name=batch_name,
            host_spec=host_spec,
            batch_script_header_file=self.batch_header,
            log=f"{output_file}.log"
        )

    def run_anaddb_file(self, ddb_file, content="", flexo=False, peizo=False):
        """
        Executes an anaddb calculation, customized for flexoelectric or piezoelectric responses.
        """
        if flexo:
            content_f = """
! anaddb calculation of flexoelectric tensor
flexoflag 1
"""
            files_content = f"""{self.file_name}_flexo_anaddb.abi
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
            command = f"anaddb < {self.file_name}_flexo_anaddb.files > {self.file_name}_flexo_anaddb.log"
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Command executed successfully: {command}")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while executing the command: {e}")
            return f"{self.file_name}_flexo_output"

        elif peizo:
            content_p = """
! Input file for the anaddb code
elaflag 3
piezoflag 3
instrflag 1
"""
            files_content = f"""
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
            command = f"anaddb < {self.file_name}_piezo_anaddb.files > {self.file_name}_piezo_anaddb.log"
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Command executed successfully: {command}")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while executing the command: {e}")
            return f"{self.file_name}_piezo_output"

        else:
            with open(f"{self.file_name}_anaddb.abi", "w") as outf:
                outf.write(content)
            command = f"anaddb < {self.file_name}_anaddb.files > {self.file_name}_anaddb.log"
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Command executed successfully: {command}")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while executing the command: {e}")

    def run_mrgddb_file(self, content):
        """
        Executes the mrgddb command using a provided input file content.
        """
        with open(f"{self.file_name}_mrgddb.in", "w") as outf:
            outf.write(content)
        command = f"mrgddb < {self.file_name}_mrgddb.in"
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Command executed successfully: {command}")
            print(f"Output: {result.stdout.decode()}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the command: {e}")
            print(f"Error output: {e.stderr.decode()}")

    def grab_energy(self, abo_file):
        """
        Retrieves the total energy from a specified Abinit output file.
        """
        if abo_file is None:
            raise Exception("Please specify the abo file you are attempting to access")
        total_energy_value = None
        try:
            with open(abo_file) as f:
                abo_content = f.read()
            match = re.search(r"total_energy\s*:\s*(-?\d+\.\d+E?[+-]?\d*)", abo_content)
            if match:
                total_energy_value = match.group(1)
                self.energy = float(total_energy_value)
            else:
                print("Total energy not found.")
        except FileNotFoundError:
            print(f"The file {abo_file} was not found.")

    def grab_flexo_tensor(self, anaddb_file=None):
        """
        Retrieves the TOTAL flexoelectric tensor from the specified file.
        """
        if anaddb_file is None:
            anaddb_file = f"file_name_energy.abo"
        flexo_tensor = None
        try:
            with open(anaddb_file) as f:
                abo_content = f.read()
            flexo_match = re.search(
                r"TOTAL flexoelectric tensor \(units= nC/m\)\s*\n\s+xx\s+yy\s+zz\s+yz\s+xz\s+xy\n((?:.*\n){9})",
                abo_content,
            )
            if flexo_match:
                tensor_strings = flexo_match.group(1).strip().split("\n")
                flexo_tensor = np.array([list(map(float, line.split()[1:])) for line in tensor_strings])
        except FileNotFoundError:
            print(f"The file {anaddb_file} was not found.")
        self.flexo_tensor = flexo_tensor

    def parse_tensor(self, tensor_str):
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
                    print(f"Could not convert line to numbers: {line}, Error: {e}")
                    raise
        return np.array(tensor_data)

    def grab_piezo_tensor(self, anaddb_file=None):
        """
        Retrieves the clamped and relaxed ion piezoelectric tensors.
        """
        if anaddb_file is None:
            anaddb_file = f"{self.file_name}_energy.abo"
        piezo_tensor_clamped = None
        piezo_tensor_relaxed = None
        try:
            with open(anaddb_file) as f:
                abo_content = f.read()
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
            print(f"The file {anaddb_file} was not found.")
        self.piezo_tensor_clamped = piezo_tensor_clamped
        self.piezo_tensor_relaxed = piezo_tensor_relaxed

    def clean_files(self, filename="filename.abi"):
        pass

    def copy_abinit_file(self):
        """
        Creates a deep copy of the current AbinitFile instance.
        """
        copied_file = copy.deepcopy(self)
        return copied_file
