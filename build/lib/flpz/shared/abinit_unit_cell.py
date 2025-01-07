from . import UnitCell
import numpy as np
import os
import re
import shutil
import sys
import tempfile
import subprocess
import time
import copy
from pathlib import Path

# To ensure the file is calling from the correct directory. 
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

class AbinitUnitCell(UnitCell):
    """
    Contains methods that allow the user to manipulate the unit cell and produce Abinit files

    Public Methods: 

    """
    # TODO: Make a method that will place all pseudopotentials into the pseudopotential folder in the program. All pp_dirpaths will then be the same

    def __init__(self, abi_file, convergence_path=None, batch_script_header_file=None):
        # Call the parent class's initializer with the keyword arguments
        self.abi_file = str(abi_file)
        self.file_name = self.abi_file.replace('.abi', '')

        super().__init__(abi_file=abi_file)

        # Convergence attributes
        self.ecut = None
        self.ecutsm = None
        self.ecutsm = None
        self.nshiftk = None
        self.shiftk = []
        self.nstep = None
        self.diemac = None
        self.ixc = None
        self.pp_dirpath = None
        self.pseudos = None
        self.kptrlatt = None
        self.nband = None
        self.toldfe = None

        # Initialize additional attributes specific to AbinitUnitCell
        if convergence_path is None: 
            self._initialize_convergence_from_file()
        self.convergence_path = convergence_path
        self.batchScriptHeader_path = batch_script_header_file
        self.runningJobs = []

        # Other attributes that need to be calculated after initialization
        self.energy = None
   


    def _initialize_convergence_from_file(self):
        """
        Extracts the convergence features from the Abinit file
        """

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Copy the contents of the original file to the temporary file
            shutil.copyfile(self.abi_file, temp_file.name)
            
            # Open the temporary file for reading
            with open(temp_file.name, 'r') as f:
                lines = f.readlines()

        # Extract ecut
        for i, line in enumerate(lines):
            if line.strip().startswith('ecut'):
                match = re.search(r"\d+", line)
                if match:
                    ecut = int(match.group())
                del lines[i]
                break

        
        self.ecut = ecut
        if self.ecut is None:
            raise Exception("ecut is missing in the Abinit file!")
        
        # Extract ecutsm
        for i, line in enumerate(lines):
            if line.strip().startswith('ecutsm'):
                match = re.search(r"\d+\.\d+|\d+", line)
                if match: 
                    ecustsm = float(match.group())
                del lines[i]
                break

        self.ecutsm = ecustsm

        # Extract nshiftk
        for i, line in enumerate(lines): 
            if line.strip().startswith('nshiftk'):
                match = re.search(r"\d+", line)
                if match:
                    nshiftk = int(match.group())
                del lines[i]
                break
        
        self.nshiftk = nshiftk
        if self.nshiftk is None:
            raise Exception("nshiftk is missing in the Abinit file!")

        # Extract shiftk
        for i, line in enumerate(lines):
            if line.strip().startswith('shiftk'):
                # Extract different features of the acell feature and map it to a float where it will be inserted into a list.
                match = re.search(r"(\d+)\*([-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*)", line)
                if match:
                    count = int(match.group(1))
                    value = float(match.group(2))
                    shiftk = [value] * count
                else:
                    shiftk = list(map(float, re.findall(r"[-+]?[0-9]*\.?[0-9]+[eE]?[-+]?[0-9]*", line)))
                # Delete extracted lines in copy
                del lines[i]

        self.shiftk = shiftk
        if self.shiftk is None:
            raise Exception("shiftk is missing in the Abinit file!")
        
        # Extract nband
        for i, line in enumerate(lines):
            if line.strip().startswith('nband'):
                match = re.search(r"\d+", line)
                if match:
                    nband = int(match.group())
                del lines[i]
                break
        
        self.nband = nband
        if self.nband is None:
            raise Exception("nband is missing in the Abinit file!")
        
        # Extract nstep 
        for i, line in enumerate(lines):
            if line.strip().startswith('nstep'):
                match = re.search(r"\d+", line)
                if match:
                    nstep = int(match.group())
                del lines[i]
                break
        
        self.nstep = nstep
        if self.nstep is None:
            raise Exception("ecut is missing in the Abinit file!")
        
        # Extract diemac 
        for i, line in enumerate(lines):
            if line.strip().startswith('diemac'):
                match = re.search(r"\d+\.\d+|\d+", line)
                if match:
                    diemac = float(match.group())
                del lines[i]
                break
        
        self.diemac = diemac
        if self.diemac is None:
            self.diemac = 4.0

        # Extract toldfe
        for i, line in enumerate(lines):
            if line.strip().startswith('toldfe'):
                match = re.search(r"toldfe\s+(\d+\.\d+d[+-]?\d+)", line)
                if match:
                    toldfe = match.group(1)
                del lines[i]
                break
        
        self.toldfe = toldfe
        
        # Extract ixc
        for i, line in enumerate(lines):
            if line.strip().startswith('ixc'):
                match = re.search(r"[-+]?\d+", line)
                if match:
                    ixc = int(match.group())
                del lines[i]
                break
        
        self.ixc = ixc
        if self.ixc is None:
            raise Exception("ixc is missing in the Abinit file!")
        
        # Extract pp_dirpath
        for i, line in enumerate(lines):
            if line.strip().startswith('pp_dirpath'):
                match = re.search(r'pp_dirpath\s+"([^"]+)"', line)
                if match: 
                    pp_dirpath = str(match.group(1))
                del lines[i]
                break
    
        self.pp_dirpath = pp_dirpath
        if self.pp_dirpath is None:
            raise Exception("pp_dirpath is missing in the Abinit file!")
        
        # Extract pseudos
        for i, line in enumerate(lines):
            if line.strip().startswith('pseudos'):
                match = re.search(r'pseudos\s+"([^"]+)"', line)
                if match: 
                    pseudos = str(match.group(1))
                del lines[i]
                break
        
        self.pseudos = pseudos
        if self.pseudos is None:
            raise Exception("pseudos is missing in the Abinit file!")
        
        # Extract kptrlatt
        kptrlatt = []
        for i, line in enumerate(lines):
            if line.strip() == 'kptrlatt':
                del lines[i]
                j = i
                while j < len(lines) and re.match(r"^\s*[-+]?\d+", lines[j]):
                    kptrlatt.append(list(map(int, lines[j].split())))
                    del lines[j]
                break
        
        self.kptrlatt = kptrlatt

        os.remove(temp_file.name)


    def findSpaceGroup(self):
        return super().findSpaceGroup()
    
    def convertToXcart(self):
        return super().convertToXcart()
    
    def convertToXred(self):
        return super().convertToXred()

    def groundWFKFile(self, output_path):
        """ 
        Creates the groundWFK.abi file with specified contents.

        Args: 
            output_path (str): path to directory to save the groundWFK.abi file.

        """
# TODO: I don't want the tolwfr to be hardcoded. I think an easy fix around is give users the ability to create their own file. 
        content = """
  ndtset 2

#Set 1 : Ground State Self-Consistent Calculation
#************************************************

  kptopt1 1
  tolvrs 1.0d-18

#Set 2 : Calculation of ddk wavefunctions
#************************************************
  kptopt2 2             # DDK can use only time reversal symmetry
  getwfk2 1             # require ground state wavefunctions from previous run
  rfelfd2 2             # activate DDK perturbation
  iscf2   -3            # this is a non-self-consistent calculation
  tolwfr2 1.0D-18       # tight convergence on wavefunction residuals
"""

        self.write_custom_abifile(output_file=output_path, header_file=content)


    def phononDispFile(self, output_path):
        """ 
        Creates the phononDispCalc.abi file with specified contents.

        Args: 
            output_path (str): path to directory to save the groundWFK.abi file.

        """
# TODO: the ngqpt is currently hardcoded and needs to be manually calculated. 

        content = """

  ndtset 6

#Definition of q-point grid
#**************************

  nqpt 1     # One qpt for each dataset
  qptopt 1
  ngqpt 4 4 4
  nshiftq 1
  shiftq 0.0 0.0 0.0

iqpt: 5 iqpt+ 1   #automatically iterate through the q pts

#Set 1 : iqpt 1 is the gamma point, so Q=0 phonons and electric field pert.
#**************************************************************************

  getddk1   98         # d/dk wave functions
  kptopt1   2          # Use of time-reversal symmetry
  rfelfd1   3          # Electric-field perturbation response
                       # (in addition to default phonon)

#Sets 2-20 : Finite-wave-vector phonon calculations (defaults for all datasets)
#******************************************************************************

   getwfk  99           # Use GS wave functions
   kptopt  3
   rfphon  1          # Do phonon response
   tolvrs  1.0d-15    # Converge on potential residual

#******
#Flags*
#******

   prtwf 1
   prtden 1
   prtpot 1
   prteig 0
"""
        self.write_custom_abifile(output_file=output_path, header_file=content)

    def write_custom_abifile(self, output_file, header_file):
        """ 
        Creates a custom Abinit file using the attributes of the class or custom files.
        
        Args: 
            output_file (str): Path to save new Abinit file
            header_file (str): The header content or path to a header file
        """
        # Determine if the header_file is actual content or a path to a file
        if "\n" in header_file or not os.path.exists(header_file):
            # If it's likely content due to newline characters or non-existent path
            header_content = header_file
        else:
            # It's a valid file path; read the content from the file
            with open(header_file, 'r') as hf:
                header_content = hf.read()

        # Write all content to the output file
        with open(f"{output_file}.abi", 'w') as outf:
            outf.write(header_content)
            
            # Append unit cell details
            outf.write("\n# Definition of unit cell")
            outf.write(f"\n#*********************************\n")
            outf.write(f"acell {' '.join(map(str, self.acell))}\n")
            outf.write(f"rprim\n")
            for coord in self.rprim:
                outf.write(f"  {'  '.join(map(str, coord))}\n")

            if self.coord_type == 'reduced':
                outf.write(f"xred\n")
            else:
                outf.write(f"xcart\n")

            for coord in self.coordinates:
                # Convert each numpy array to a flat list
                outf.write(f"  {'  '.join(map(str, coord))}\n")

            outf.write("\n# Definition of atoms")
            outf.write(f"\n#*********************************\n")
            outf.write(f"natom {self.num_atoms} \n")
            outf.write(f"ntypat {self.ntypat} \n")
            outf.write(f"znucl {' '.join(map(str, self.znucl))}\n")
            outf.write(f"typat {' '.join(map(str, self.typat))}\n")

            if self.convergence_path is None:
                outf.write("\n# Definition of the planewave basis set")
                outf.write(f"\n#*********************************\n")
                outf.write(f"ecut {self.ecut} \n")
                if self.ecutsm is not None:
                    outf.write(f"ecutsm {self.ecutsm} \n")

                outf.write(f"\n# Definition of the k-point grid")
                outf.write(f"\n#********************************* \n")
                outf.write(f"nshiftk {self.nshiftk} \n")
                if self.kptrlatt is not None:
                    for i in self.kptrlatt:
                        outf.write(f"  {' '.join(map(str, i))}\n")
                outf.write(f"shiftk {' '.join(map(str, self.shiftk))} \n")
                outf.write(f"nband {self.nband} \n")
                outf.write("\n# Definition of the SCF Procedure")
                outf.write(f"\n#********************************* \n")
                outf.write(f"nstep {self.nstep} \n")
                outf.write(f"diemac {self.diemac} \n")
                outf.write(f"ixc {self.ixc} \n")
                outf.write(f"toldfe {self.toldfe}\n")
                outf.write(f"\npp_dirpath \"{self.pp_dirpath}\" \n")
                outf.write(f"pseudos \"{self.pseudos}\" \n")
                print(f"{output_file} was created successfully!")
            else:
                with open(self.convergence_path, 'r') as cf:
                    convergence_content = cf.read()
                outf.write(convergence_content)


                
    def all_jobs_finished(self):
            """
            Checks if all jobs in self.runningJobs have finished.

            Returns:
                bool: True if all jobs are finished, False otherwise.
            """
            for job_id in self.runningJobs:
                # Check the status of the specific job
                result = subprocess.run(['squeue', '-j', str(job_id)], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Error checking job {job_id} status:", result.stderr)
                    continue
                
                # If the job is found in the queue, it means it's still running or pending
                if str(job_id) in result.stdout:
                    return False
            
            # If none of the jobs were found in the queue, they have all finished
            return True
    
    def wait_for_jobs_to_finish(self, check_time=60):
        """
        Waits until all jobs in self.runningJobs are finished.
        """
        print("Waiting for jobs to finish...")
        while not self.all_jobs_finished():
            print(f"Jobs still running. Checking again in {check_time} seconds...")
            time.sleep(check_time)
        print("All jobs have finished.")
        self.runningJobs = []

    def run_abinit(self, input_file='abinit', batch_name='abinit_job',
                batch_script_header_file=None, host_spec='mpirun -hosts=localhost -np 30', 
                delete_batch_script=True, log="log"):
        """
        Run the Abinit program using the generated input file.

        Args:
            input_file (str): The abinit file that will be executed.
            batch_script_header_file (str): The header of the batch script.
            host_spec (str): The command to be run in the batch script.
            delete_batch_script (bool): Option to delete the batch script after execution.
            log (str): Log file for output.
        """

        
        # Compile the content to be written in the file
        content = f"""{input_file}.abi
{input_file}.abo
{input_file}_generic_input_files
{input_file}_generic_output_files
{input_file}_generic_temp_files
    """

        if batch_script_header_file is not None:
            # Create a non-temporary file in the current directory
            file_path = f"{input_file}_abinit_input_data.txt"
            
            with open(file_path, 'w') as file:
                file.write(content)
            try:
                # Use the regular file's path in your operations
                script_created = self.write_batch_script(batch_script_header_file=batch_script_header_file, 
                                                        input_file=file_path, 
                                                        batch_name=f"{batch_name}.sh",
                                                        host_spec=host_spec, 
                                                        log=log)
                print(f"Was the batch script successfully created: {script_created}")

                # Submit the job using subprocess to capture output
                result = subprocess.run(['sbatch', f"{batch_name}.sh"], capture_output=True, text=True)

                if result.returncode == 0:
                    print("Batch job submitted using 'sbatch'.")
                    try:
                        job_number = int(result.stdout.strip().split()[-1])
                        self.runningJobs.append(job_number)
                        print(f"Job number {job_number} added to running jobs.")
                    except (ValueError, IndexError) as e:
                        print(f"Failed to parse job number: {e}")
                else:
                    print("Failed to submit batch job:", result.stderr)

            finally:
                # TODO: Write a method that cleans files for you
                print("Attempt to delete")

        else:
            command = f"abinit < {input_file} > {log}"
            os.system(command)
            print(f"Abinit executed directly. Output written to '{log}'.")



    def write_batch_script(self, batch_script_header_file='default_batch_file', input_file='input.in', batch_name='default_output', host_spec=None, log='log'):
        """
        Writes a batch script using a prewritten header file 

        Args: 
            batch_script_header_file (str): The file containing the header of the batch script.
            input_file (str): The name of the input file fed into Abinit.
            batch_name (str): The name of the created batch script.
            host_spec (str): The command to be run in the batch script.
            log (str): The name of the log file.
        """

        # Read the content from the batch_script_header file
        try:
            with open(batch_script_header_file, 'r') as header_file:
                batch_script_header = header_file.read()
        except FileNotFoundError:
            print(f"Error: The file {batch_script_header_file} does not exist.")
            return False
        except Exception as e:
            print(f"An error occurred while reading the batch script header file: {e}")
            return False

        # Write to the output file
        try:
            with open(f"{batch_name}", 'w') as file:
                # Write the contents of the batch script header
                file.write("#!/bin/bash\n")
                file.write(batch_script_header)
                
                if host_spec is None:
                    file.write(f"\nmpirun -np 8 abinit < {input_file} > {log} \n")
                else:
                    file.write(f"\n{host_spec} abinit < {input_file} > {log} \n")

            print("Batch script was written successfully.")
            return True

        except Exception as e:
            print(f"An error occurred while writing the batch script: {e}")
            return False
        
    def perturbations(self, pert):
        """
        Apply a given perturbation to the unit cell coordinates and return a new UnitCell.
        
        Args:
            pert (np.ndarray): A numpy array representing the perturbation to be applied.

        Returns:
            UnitCell: A new instance of UnitCell with perturbed coordinates.
        """

        # Ensure the perturbation has the correct shape
        perturbation = np.array(pert, dtype=float)
        if perturbation.shape != self.coordinates.shape:
            raise ValueError("Perturbation must have the same shape as the coordinates.")

        copy_cell = self.copy_abinit_unit_cell()
        # Calculate new coordinates by adding the perturbation
        new_coordinates = self.coordinates + perturbation

        # Create a new UnitCell object with the updated coordinates
        # Assuming other properties remain the same; adjust as needed
        copy_cell.coordinates = new_coordinates

        return copy_cell

    def copy_abinit_unit_cell(self):
        """
        Makes a deep copy of an AbinitUnitCell instance.

        Args:
            original_cell (AbinitUnitCell): The original AbinitUnitCell instance to be copied

        Returns:
            AbinitUnitCell: A new intance that is a deep copy of the original. 
        """

        # Perform a deep copy to ensure all nested objects are also copied
        copied_cell = copy.deepcopy(self)
        return copied_cell
    
    def run_energy_calculation(self, host_spec='mpirun -hosts=localhost -np 30'):
        """
        Run an energy calculation 
        """
        content = f"""ndtset 1

    # Ground State Self-Consistency
    #*******************************

    getwfk1 0
    kptopt1 1
    tolvrs 1.0d-18

    # turn off various file outputs
    prtpot 0
    prteig 0


    getwfk 1
    useylm 1  # Use of spherical harmonics
    kptopt 2  # Takes into account time-reversal symmetry. 

    """
        # Get the current working directory
        working_directory = os.getcwd()
        
        # Construct the full paths for the output and batch files
        output_file = os.path.join(working_directory, f"{self.file_name}_energy")
        batch_name = os.path.join(working_directory, f"{self.file_name}_bscript")
        
        # Use these paths in your methods
        self.write_custom_abifile(output_file=output_file, header_file=content)
        self.run_abinit(input_file=output_file, batch_name=batch_name, host_spec=host_spec)

    def grab_energy(self, abo_file=None):
        """
        Grab the energy of an abo file. 
        """
        if abo_file is None:
            abo_file = f"{self.file_name}_energy.abo"

        # Ensure total_energy_value is initialized
        total_energy_value = None
        
        try:
            with open(abo_file) as f:
                # Read all content as a single string
                abo_content = f.read()

            # Apply the regex pattern to the full content
            match = re.search(r"total_energy\s*:\s*(-?\d+\.\d+E?[+-]?\d*)", abo_content)
            
            if match:
                total_energy_value = match.group(1)
            else:
                print("Total energy not found.")
                
        except FileNotFoundError:
            print(f"The file {abo_file} was not found.")
        
        self.energy = total_energy_value


    def change_coordinates(self, new_coordinates, cartesian=False, reduced=False):
        """
        Change the coordinates of the unit cell
        """
        self.coordinates = new_coordinates
        self.energy = None
        if cartesian == True: 
            self.coord_type = 'cartesian'
        elif reduced == True:
            self.coord_type = 'reduced'
        
