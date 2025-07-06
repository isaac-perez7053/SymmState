import os
import copy
import numpy as np
from symmstate.abinit.abinit_structure import AbinitStructure
from symmstate.pseudopotentials.pseudopotential_manager import PseudopotentialManager
from symmstate.templates import TemplateManager
from symmstate.utils import get_unique_filename
from typing import Optional, List, Dict
from symmstate.slurm import SlurmFile


class AbinitFile(AbinitStructure):
    """
    Class for writing, managing, and executing Abinit input files (.abi).

    This class extends AbinitUnitCell to provide:
      - Creation and management of .abi files.
      - Execution of Abinit jobs either via SLURM batch submission or direct OS calls.
      - Support for symmetry-adapted modes basis through smodes_input and target_irrep.
      - Integrated logging and explicit type handling.

    The AbinitFile can be initialized using an existing .abi file, a Structure object, or
    symmetry-adapted input parameters.
    """

    def __init__(
        self,
        abi_file: str = None,
    ) -> "AbinitFile":
        """
        Initialize an AbinitFile instance using an Abinit file or dictionary

        Parameters:
            abi_file (str):
                Path to an existing Abinit file. If provided, the file name (without .abi extension)
                is used for further processing.

        """
        # Initialize AbinitUnitCell with supported parameters.
        obj = AbinitStructure.from_file(abi_file)
        for attr, value in vars(obj).items():
            setattr(self, attr, value)
        self.filename = str(abi_file).replace(".abi", "")

    @classmethod
    def from_dict(cls, dict: Dict, filename: str) -> "AbinitFile":
        """
        Initialize from a dictionary and filename.
        """
        obj = cls(dict)
        obj.filename = filename
        return obj

    def write(
        self,
        output_file: str,
        content: str,
        pseudos: List = [],
    ) -> str:
        """
        Write a custom Abinit .abi file with a header and simulation parameters.

        This function writes an Abinit input file using the given header (either as text or read
        from a file) and appends simulation parameters (unit cell, coordinates, atoms, basis set,
        k-point grid, SCF settings, and pseudopotentials). It generates a unique filename to prevent
        conflicts.

        Parameters:
            output_file (str):
                Base filename for the output file; a unique name is generated.
            content (str):
                Header content as a literal string or a file path to be read.
            pseudos (List, optional):
                List of pseudopotential identifiers; if empty, defaults to those in self.vars.

        Returns:
            str: The unique output file name (without the .abi extension).
        """
        # Check input_file has .abi extension. If it does, get rid of it
        output_file = output_file.replace(".abi", "")

        # Determine whether 'content' is literal text or a file path.
        if "\n" in content or not os.path.exists(content):
            header_content: str = content
        else:
            with open(content, "r") as hf:
                header_content = hf.read()

        # Generate a unique filename.
        output_file = get_unique_filename(output_file)

        with open(f"{output_file}.abi", "w") as outf:
            # Write header
            outf.write(header_content)
            outf.write("\n#--------------------------\n"
                    "# Definition of unit cell\n"
                    "#--------------------------\n")

            # Lattice parameters
            acell = self.vars.get("acell", self.lattice.abc)
            outf.write(f"acell {' '.join(map(str, acell))}\n")

            rprim = self.vars.get("rprim", self.lattice.matrix.tolist())
            outf.write("rprim\n")
            for row in rprim:
                outf.write(f"  {'  '.join(map(str, row))}\n")

            # Atomic coordinates
            coord_key = self.vars['coord_type']
            outf.write(f"{coord_key}\n")
            coordinates = self.vars['coordinates']
            for row in coordinates:
                outf.write(f"  {'  '.join(map(str, row))}\n")

            # Atomic information
            outf.write("\n#--------------------------\n"
                    "# Definition of atoms\n"
                    "#--------------------------\n")
            outf.write(f"natom {self.vars['natom']}\n")
            outf.write(f"ntypat {self.vars['ntypat']}\n")
            outf.write(f"znucl {' '.join(map(str, self.vars['znucl']))}\n")
            outf.write(f"typat {' '.join(map(str, self.vars['typat']))}\n")

            # Plane wave basis
            outf.write("\n#----------------------------------------\n"
                    "# Definition of the planewave basis set\n"
                    "#----------------------------------------\n")
            if self.vars.get("ecut") is not None:
                outf.write(f"ecut {self.vars.get('ecut', "")}\n")
            if self.vars.get("ecutsm") is not None:
                outf.write(f"ecutsm {self.vars['ecutsm']}\n")

            # K-point grid
            outf.write("\n#--------------------------\n"
                    "# Definition of the k-point grid\n"
                    "#--------------------------\n")
            kptrlatt = self.vars.get("kptrlatt")
            if kptrlatt is not None:
                outf.write("kptrlatt\n")
                for row in kptrlatt:
                    outf.write("  " + " ".join(map(str, row)) + "\n")
            elif self.vars.get("ngkpt") is not None:
                outf.write(f"ngkpt {' '.join(map(str, self.vars['ngkpt']))}\n")
            outf.write(f"nshiftk {self.vars.get('nshiftk', 1)}\n")
            shiftk = self.vars.get("shiftk", [[0.5, 0.5, 0.5]])
            shiftk_arr = np.array(shiftk)

            outf.write("shiftk\n")
            if shiftk_arr.ndim == 1:
                outf.write("  " + " ".join(map(str, shiftk_arr.tolist())) + "\n")
            else:
                for row in shiftk_arr:
                    outf.write("  " + " ".join(map(str, row.tolist())) + "\n")

            outf.write(f"nband {self.vars['nband']}\n")

            # SCF procedure
            outf.write("\n#--------------------------\n"
                    "# Definition of the SCF Procedure\n"
                    "#--------------------------\n")
            outf.write(f"nstep {self.vars.get('nstep', 9)}\n")
            outf.write(f"diemac {self.vars.get('diemac', '1000000.0')}\n")
            outf.write(f"ixc {self.vars['ixc']}\n")

            conv_key = self.vars['conv_criteria']
            outf.write(f"{conv_key} {self.vars[conv_key]}\n")

            # Pseudopotentials
            pp_dir_path = PseudopotentialManager().folder_path
            outf.write(f'\npp_dirpath "{pp_dir_path}"\n')

            if not pseudos:
                pseudos = self.vars.get("pseudos", [])
            pseudo_str = ", ".join(pseudos).replace('"', "")
            outf.write(f'pseudos "{pseudo_str}"\n')

            print(f"The Abinit file {output_file}.abi was created successfully!")

        return f"{output_file}.abi"

    def execute(
        self,
        input_file: str,
        slurm_obj: Optional[SlurmFile],
        *,
        batch_name: Optional[str] = None,
        log_file: Optional[str] = None,
        extra_commands: Optional[str] = None,
    ) -> str:
        """
        execute the Abinit simulation via batch submission or direct execution.

        If a SlurmFile is provided, a unique input data file and batch script are created,
        and the job is submitted. Otherwise, the Abinit command is executed directly with
        output redirected to the specified log file.

        Parameters:
            input_file (str): Base name for the Abinit input files.
            slurm_obj (Optional[SlurmFile]): Object for managing batch operations; if None, execute directly.
            batch_name (Optional[str], keyword-only): Custom name for the batch script.
            log_file (Optional[str], keyword-only): Path to the log file.
            extra_commands (Optional[str], keyword-only): Additional commands for the batch script.

        Returns:
            None

        Raises:
            Exception: If batch script creation or submission fails.
        """
        # Check input_file has .abi extension. If it does, get rid of it
        input_file = input_file.replace(".abi", "")
        if batch_name is None:
            batch_name = f"{input_file}_batch.sh"

        if log_file is None:
            log_file = f"{input_file}.log"

        content: str = f"""{input_file}.abi
{input_file}.abo
{input_file}o
{input_file}_gen_output
{input_file}_temp
        """
        # We now require a SlurmFile object (self.slurm_obj) to handle batch script operations.
        if slurm_obj is not None:
            file_path: str = f"{input_file}.files"
            file_path = get_unique_filename(file_path)
            with open(file_path, "w") as file:
                file.write(content)
            try:

                # Use the provided SlurmFile object.
                script_created = slurm_obj.write_batch_script(
                    input_file=f"{input_file}.abi",
                    log_file=log_file,
                    batch_name=batch_name,
                    extra_commands=extra_commands,
                )
                print(f"Batch script created: {script_created}")
                slurm_obj.submit_job(script_created)

            except Exception as e:
                print(f"Failed to execute abinit using the batch script: {e}")
                raise  

        else:
            # If no SlurmFile object was provided, execute directly using extra_commands feature
            os.system(extra_commands)
            print(f"Abinit executed directly. Output written to '{log_file}'.")

        return f"{input_file}.abi"

    def execute_piezo_calculation(
        self,
        *,
        output_name: Optional[str] = None,
        slurm_obj: Optional[SlurmFile] = None,
        batch_name: Optional[str] = None,
        log_file: Optional[str] = None,
        extra_commands: Optional[str] = None,
    ) -> str:
        """
        execute a piezoelectricity calculation for the unit cell.

        This function creates a custom Abinit input file with predefined settings for
        a piezoelectric calculation and then executes Abinit either via a batch job (if a
        SlurmFile is provided) or directly. The function returns the base name of the
        generated output file.

        Parameters:
            slurm_obj (Optional[SlurmFile]):
                An object for managing batch job submission; if None, the calculation is execute directly.
            log_file (Optional[str], keyword-only):
                Path to the log file where output from the Abinit execute is saved.
            extra_commands (Optional[str], keyword-only):
                Additional commands to be included in the batch script.

        Returns:
            str: The unique base name of the output file used for the piezoelectric calculation.
        """
        content: str = TemplateManager().unload_special_template("_piezoelectric_script")
        working_directory: str = os.getcwd()

        # Preset name if user didn't choose one
        if output_file is None:
            output_name = f"{self.filename}_piezo.abi"
            output_name = get_unique_filename(output_name)

        output_file: str = os.path.join(working_directory, output_name)
        output_file = get_unique_filename(output_file)
        output_file = self.write(
            output_file=output_file, content=content, coords_are_cartesian=False
        )
        self.execute(
            input_file=output_file,
            slurm_obj=slurm_obj,
            batch_name=batch_name,
            log_file=log_file,
            extra_commands=extra_commands,
        )

        return output_file

    def execute_flexo_calculation(
        self,
        *,
        output_file: Optional[str] = None,
        slurm_obj: Optional[SlurmFile],
        batch_name: Optional[str] = None,
        log_file: Optional[str] = None,
        extra_commands: Optional[str] = None,
    ) -> str:
        """
        execute a flexoelectricity calculation for the unit cell.

        This function generates a custom Abinit input file with settings for a flexoelectricity
        calculation and then executes the calculation via a batch job using the provided
        SlurmFile object.

        Parameters:
            slurm_obj (SlurmFile):
                An object to manage batch submission and related operations.
            log_file (Optional[str], keyword-only):
                Path to the file where the calculation log will be stored.
            extra_commands (Optional[str], keyword-only):
                Additional commands to include in the batch script.

        Returns:
            str: The base name of the generated output file (without extension).

        """

        content: str = TemplateManager().unload_special_template("_flexoelectric_script")
        working_directory: str = os.getcwd()

        # Generate file name if note specified 
        if output_file is None:
            output_name = f"{self.filename}_flexo.abi"
            output_name = get_unique_filename(output_name)

        output_file: str = os.path.join(working_directory, output_name)
        output_file = get_unique_filename(output_file)
        output_file = self.write(
            output_file=output_file, content=content, coords_are_cartesian=False
        )
        self.execute(
            input_file=output_file,
            slurm_obj=slurm_obj,
            batch_name=batch_name,
            log_file=log_file,
            extra_commands=extra_commands,
        )

        return output_file

    def execute_energy_calculation(
        self,
        *,
        output_file: Optional[str] = None,
        slurm_obj: Optional[SlurmFile],
        batch_name: Optional[str] = None,
        log_file: Optional[str] = None,
        extra_commands: Optional[str] = None,
    ) -> str:
        """
        execute an energy calculation for the unit cell.

        This function generates a custom Abinit input file configured for an energy
        calculation and executes it via the provided SlurmFile for batch submission.

        Parameters:
            slurm_obj (SlurmFile):
                Object used for batch job submission.
            log_file (Optional[str], keyword-only):
                Path to the log file to capture output.
            extra_commands (Optional[str], keyword-only):
                Additional commands to include in the batch script.

        Returns:
            str: The base name of the generated output file.
        """

        # Grab template for energy calculation
        content: str = TemplateManager().unload_special_template("_energy_script")
        working_directory: str = os.getcwd()

        # Generate pre-made name 
        if output_file is None:
            output_name = f"{self.filename}_energy.abi"
            output_name = get_unique_filename(output_name)

        output_file: str = os.path.join(working_directory, output_name)
        output_file = self.write(
            output_file=output_file, content=content, coords_are_cartesian=True
        )
        self.execute(
            input_file=output_file,
            slurm_obj=slurm_obj,
            batch_name=batch_name,
            log_file=log_file,
            extra_commands=extra_commands,
        )
        return output_file
    
    def copy(self):
        """
        Creates a deep copy of the current AbinitFile instance.

        Returns:
            AbinitFile: A new instance that is a deep copy of self.
        """
        return copy.deepcopy(self)


    def __repr__(self):

        lines = []
        lines.append("#--------------------------")
        lines.append("# Definition of unit cell")
        lines.append("#--------------------------")
        acell = self.vars.get("acell", self.lattice.abc)
        lines.append(f"acell {' '.join(map(str, acell))}")
        rprim = self.vars.get("rprim", self.lattice.matrix.tolist())
        lines.append("rprim")
        for coord in rprim:
            lines.append(f"  {'  '.join(map(str, coord))}")
        # Choose coordinate system: xcart if available; otherwise xred.
        if self.vars.get("xcart") is not None:
            lines.append("xcart")
        else:
            lines.append("xred")
        coordinates = self.vars["coordinates"]
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
        if self.vars.get("kptrlatt") is not None:
            kptrlatt = self.vars["kptrlatt"]
            lines.append("kptrlatt")
            for row in kptrlatt:
                lines.append(f"  {'  '.join(map(str, row))}")
        elif self.vars.get("ngkpt") is not None:
            lines.append(f"ngkpt {' '.join(map(str, self.vars['ngkpt']))} \n")
        # Make sure to split shiftk if it's a string
        lines.append(f"nshiftk {self.vars.get('nshiftk', '1')}")
        shiftk = self.vars.get("shiftk", [[0.5, 0.5, 0.5]])
        shiftk_arr = np.array(shiftk)

        lines.append("shiftk")
        if shiftk_arr.ndim == 1:
            lines.append("  " + "  ".join(map(str, shiftk_arr.tolist())))
        else:
            for row in shiftk_arr:
                lines.append("  " + "  ".join(map(str, row.tolist())))
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
    
