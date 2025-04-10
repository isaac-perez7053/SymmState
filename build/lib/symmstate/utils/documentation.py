import textwrap

class Documentation:
    def __init__(self):
        self.define_documentation()

    def define_documentation(self):
        self.UNITCELL =  textwrap.dedent(r"""UnitCell Class
        ==============
        Defines the UnitCell class which contains all the necessary information for a unit cell used in solid state calculations.

        Overview
        --------
        The UnitCell class extends SymmStateCore and encapsulates all the necessary information for a unit cell.
        Users may specify the cell either by manually providing lattice parameters, primitive vectors, and atomic coordinates (or via a complete pymatgen Structure), or by providing a SMODES file along with a target irreducible representation.
        In the latter mode the SMODES output is used to build the cell (manual parameters are ignored).

        Initialization
        --------------
        There are two modes to initialize a UnitCell:

        1. **Manual Mode**  
        Provide the parameters:
        - **acell:** The lattice scaling factors.
        - **rprim:** The primitive lattice vectors (as an array).
        - **coordinates:** The atomic coordinates.
        - **coords_are_cartesian:** A boolean indicating whether coordinates are cartesian (True) or reduced (False).
        - **elements:** The element symbols for each site.

        Alternatively, pass a pre-built pymatgen Structure object using the `structure` parameter.

        2. **SMODES Mode**  
        Supply the `smodes_file` and `target_irrep` parameters. In this mode the SMODES output is used to compute all structural parameters using the symmetry‐adapted basis.

        Initialization Parameters
        -------------------------
        - **structure** (optional pymatgen.core.Structure): A pre-built structure.
        - **acell** (optional list of float): Lattice scaling factors.
        - **rprim** (optional numpy.ndarray): Primitive lattice vectors.
        - **coordinates** (optional numpy.ndarray): Atomic coordinates.
        - **coords_are_cartesian** (optional bool): Whether coordinates are in cartesian form.
        - **elements** (optional list of str): List of element symbols.
        - **smodes_file** (optional str): Path to a SMODES input file.
        - **target_irrep** (optional str): Target irreducible representation.
        - **symm_prec** (float, default=1e-5): Tolerance for symmetry calculations.

        Public Methods
        --------------
        - **grab_reduced_coordinates()**  
        Returns a copy of the unit cell’s fractional coordinates.

        - **grab_cartesian_coordinates()**  
        Returns a copy of the unit cell’s cartesian coordinates.

        - **find_space_group()**  
        Uses pymatgen’s SpacegroupAnalyzer to compute and return the space group (number and symbol).

        - **perturbations(perturbation, coords_are_cartesian=False)**  
        Applies a given perturbation to the cell coordinates and returns a new UnitCell.  
        *Args:*  
        - **perturbation (np.ndarray):** Displacement to add.  
        - **coords_are_cartesian (bool):** If True, add to cartesian coordinates; if False, to fractional coordinates.  
        *Returns:* A new UnitCell instance with perturbed coordinates.

        Internal Methods
        ----------------
        - **_round_to_nearest(value)**: Rounds a floating-point value to a precision of 1e-15.
        - **clean_reduced_coordinates()**: Cleans the fractional coordinates by removing numerical noise.

        Usage Examples
        --------------
        **Manual Construction Example:**

        .. code-block:: python

            from pymatgen.core import Structure, Lattice
            from symmstate.unit_cell_module import UnitCell

            acell = [1.0, 1.0, 1.0]
            rprim = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
            coordinates = [[0, 0, 0],
                        [0.5, 0.5, 0.5]]
            elements = ["Na", "Cl"]

            cell = UnitCell(acell=acell, rprim=rprim, coordinates=coordinates, 
                            coords_are_cartesian=False, elements=elements)

            print("Fractional Coordinates:", cell.grab_reduced_coordinates())
            print("Space Group:", cell.find_space_group())

        **SMODES Construction Example:**

        .. code-block:: python

            cell = UnitCell(smodes_file="path/to/smodes_input.txt", target_irrep="GM4-", symm_prec=1e-5)
            print("Structure determined via SMODES:")
            print(cell)

        Additional Information
        ----------------------
        The UnitCell class automatically cleans its numerical precision issues via the clean_reduced_coordinates() method, ensuring high consistency in further calculations.
        """)

        # Extended documentation for the AbinitUnitCell class in Markdown.
        self.ABINIT_UNIT_CELL_DOC = textwrap.dedent(r"""
        AbinitUnitCell Class Documentation
        ==================================

        Overview
        --------
        The **AbinitUnitCell** class extends **UnitCell** with Abinit‑specific functionality and initialization paths.
        It allows you to either:
        - Initialize a unit cell manually by providing lattice parameters, primitive vectors, coordinates, and a list of element symbols, or
        - Provide a SMODES input file along with a target irreducible representation so that the symmetry‑adapted basis is computed (any manually provided parameters are then ignored).

        Initialization
        --------------
        The AbinitUnitCell can be initialized in two ways:

        1. **Manual Mode:**  
        Pass the following parameters:
        - **acell:** A list of lattice scaling factors.
        - **rprim:** A NumPy array (or list of lists) representing the primitive lattice vectors.
        - **coordinates:** A NumPy array (or list of lists) containing the atomic positions.
        - **coords_are_cartesian:** Boolean flag specifying if the coordinates are cartesian (`True`) or fractional (`False`).
        - **elements:** A list of element symbols corresponding to the atomic sites.

        2. **SMODES Mode:**  
        Pass the `smodes_file` and `target_irrep` parameters along with an optional `symm_prec` tolerance. In this mode, the SMODES output is used to generate the unit cell, and any manually provided parameters are overridden.

        Public Methods
        --------------
        - **update_unit_cell_parameters():**  
        Updates internal parameters (e.g. number of atoms, sorted atomic numbers, typat, nband) based on the current structure while preserving the original ordering of atomic sites.

        - **_derive_abinit_parameters():**  
        Derives and stores key structural parameters (e.g. acell, rprim, znucl, typat, natom, ntypat) from the structure.

        - **_init_from_abinit_vars():**  
        Initializes a new UnitCell from the parameters stored in the `vars` attribute.

        - **_convert_znucl_typat():**  
        Converts the numerical atomic numbers (from `znucl`) and type assignments (`typat`) to a list of element symbols using pymatgen’s Element class.

        - **copy_abinit_unit_cell():**  
        Creates a deep copy of the instance (useful if you need an independent copy of the cell).

        - **change_coordinates(new_coordinates, coords_are_cartesian=False):**  
        Updates the cell’s coordinates and the corresponding entries in the variable dictionary.

        - **perturbations(perturbation, coords_are_cartesian=False):**  
        Applies a given numerical perturbation (as a NumPy array) to the current cell’s coordinates and returns a new AbinitUnitCell instance with the modified structure.

        - **grab_reduced_coordinates() / grab_cartesian_coordinates():**  
        Return copies of the cell’s fractional and cartesian coordinates, respectively.

        - **find_space_group():**  
        Uses pymatgen’s SpacegroupAnalyzer to compute and return the space group information (both the number and symbol).

        Internal Methods
        ----------------
        - **_process_atoms(atom_list):**  
        Computes the total number of atoms, unique atomic numbers, type assignments (typat), and returns these values.

        - **clean_reduced_coordinates():**  
        Cleans the fractional coordinates by removing minor numerical noise via rounding.

        Usage Examples
        --------------
        **Manual Construction Example:**

        ```python
        from pymatgen.core import Structure, Lattice
        from symmstate.unit_cell_module import UnitCell
        from symmstate.abinit_unit_cell import AbinitUnitCell

        acell = [1.0, 1.0, 1.0]
        rprim = [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]
        coordinates = [[0, 0, 0],
                    [0.5, 0.5, 0.5]]
        elements = ["Na", "Cl"]

        cell = AbinitUnitCell(acell=acell, rprim=rprim, coordinates=coordinates,
                            coords_are_cartesian=False, elements=elements)
        print("Reduced Coordinates:", cell.grab_reduced_coordinates())
        print("Space Group:", cell.find_space_group())""")


        self.ABINIT_UNIT_CELL_DOC = r"""
        AbinitUnitCell Class Documentation
        ==================================

        Overview
        --------
        The AbinitUnitCell class extends UnitCell with Abinit-specific functionality and initialization paths.
        It allows you to either:
        - Initialize a unit cell manually by providing lattice parameters, primitive vectors,
            coordinates, and a list of element symbols, or
        - Provide a SMODES input file along with a target irreducible representation so that
            the symmetry-adapted basis is computed (any manually provided parameters are then ignored).

        Initialization
        --------------
        The AbinitUnitCell can be initialized in two ways:

        1. **Manual Mode:**  
        Pass the following parameters:
        - **acell:** A list of lattice scaling factors.
        - **rprim:** A NumPy array (or list of lists) representing the primitive lattice vectors.
        - **coordinates:** A NumPy array (or list of lists) containing the atomic positions.
        - **coords_are_cartesian:** Boolean flag specifying if the coordinates are cartesian (True) or fractional (False).
        - **elements:** A list of element symbols corresponding to the atomic sites.

        2. **SMODES Mode:**  
        Pass the `smodes_file` and `target_irrep` parameters along with an optional
        `symm_prec` tolerance. In this mode, the SMODES output is used to generate the unit cell,
        and the manual parameters provided (if any) are overridden.

        Public Methods
        --------------
        - **update_unit_cell_parameters():**  
        Updates the internal parameters (e.g., number of atoms, sorted atomic numbers, typat, nband)
        based on the current structure while preserving the original ordering of atomic sites.

        - **_derive_abinit_parameters():**  
        Derives and stores key structural parameters (e.g., acell, rprim, znucl, typat, natom, ntypat)
        from the structure.

        - **_init_from_abinit_vars():**  
        Initializes a new UnitCell from the parameters stored in the `vars` attribute.
        
        - **_convert_znucl_typat():**  
        Converts the numerical atomic numbers (from `znucl`) and type assignments (`typat`) to a list
        of element symbols using pymatgen’s Element class.

        - **copy_abinit_unit_cell():**  
        Creates a deep copy of the instance (useful if you need an independent copy of the cell).

        - **change_coordinates(new_coordinates, coords_are_cartesian=False):**  
        Updates the unit cell’s coordinates and the corresponding entries in the variable dictionary.

        - **perturbations(perturbation, coords_are_cartesian=False):**  
        Applies a given numerical perturbation (as a NumPy array) to the current cell’s coordinates and returns
        a new AbinitUnitCell instance with the modified structure.

        - **grab_reduced_coordinates() / grab_cartesian_coordinates():**  
        Return copies of the cell’s fractional and cartesian coordinates, respectively.

        - **find_space_group():**  
        Utilizes pymatgen’s SpacegroupAnalyzer to determine and return the space group information
        (including both the space group number and symbol).

        Internal Methods
        ----------------
        - **_process_atoms(atom_list):**  
        Computes the total number of atoms, unique atomic numbers, type assignments (typat), and returns these values.

        - **clean_reduced_coordinates():**  
        Cleans the fractional coordinates by removing minor numerical noise via rounding.

        Usage Examples
        --------------
        **Manual Construction Example:**

        .. code-block:: python

            from pymatgen.core import Structure, Lattice
            from symmstate.unit_cell_module import UnitCell
            from symmstate.abinit_unit_cell import AbinitUnitCell

            acell = [1.0, 1.0, 1.0]
            rprim = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
            coordinates = [[0, 0, 0],
                        [0.5, 0.5, 0.5]]
            elements = ["Na", "Cl"]

            cell = AbinitUnitCell(acell=acell, rprim=rprim, coordinates=coordinates,
                                coords_are_cartesian=False, elements=elements)
            print("Reduced Coordinates:", cell.grab_reduced_coordinates())
            print("Space Group:", cell.find_space_group())

        **SMODES Mode Example:**

        .. code-block:: python

            cell = AbinitUnitCell(smodes_file="path/to/smodes_input.txt", target_irrep="GM4-", symm_prec=1e-5)
            print("Structure determined via SMODES:")
            print(cell)
        """

        self.ABINIT_FILE_DOC = r"""
        AbinitFile Class Documentation
        ================================

        Overview
        --------
        The **AbinitFile** class extends the AbinitUnitCell class with additional functionality for writing and executing
        Abinit input files. It provides a framework to generate custom `.abi` files, handle unit cell definitions,
        and interface with a Slurm queue system for batch job submission.

        Initialization
        --------------
        AbinitFile can be initialized in one of these ways:
        - **Using an existing Abinit file:**  
            Provide the path through the `abi_file` parameter.
        - **From a pymatgen Structure:**  
            Provide a Structure object via the `unit_cell` parameter.
        - **Via SMODES mode:**  
            Provide both a `smodes_input` file and a `target_irrep`; in this mode, the SMODES output is used to generate
            the symmetry-adapted basis, and any manual parameters are ignored.
        - Optionally, you can provide a SlurmFile object (`slurm_obj`) for managing batch job submissions.

        Public Methods
        --------------
        - **write_custom_abifile(output_file, content, coords_are_cartesian, pseudos):**  
        Writes a custom Abinit input file (.abi) by outputting header information, unit cell parameters,
        atomic coordinates, and other input settings. If `content` is not a multi‑line string,
        it is interpreted as a file path.

        - **run_abinit(input_file, batch_name, host_spec, delete_batch_script, log):**  
        Executes the Abinit program by using the generated input file. If a SlurmFile object is provided,
        it creates a batch script and submits the job via `sbatch`; otherwise, it runs Abinit directly.

        - **run_piezo_calculation(host_spec):**  
        Generates an input file and executes a piezoelectricity calculation.

        - **run_flexo_calculation(host_spec):**  
        Generates an input file and executes a flexoelectricity calculation.

        - **run_energy_calculation(host_spec):**  
        Generates an input file and executes an energy calculation.

        - **run_anaddb_file(content, files_content, ddb_file, flexo, peizo):**  
        Invokes an anaddb post-processing calculation to evaluate either the flexoelectric or piezoelectric responses.

        - **copy_abinit_file():**  
        Returns a deep copy of the current AbinitFile object.

        - **__repr__():**  
        Returns a formatted string representing the Abinit file contents, including unit cell, atom definitions,
        basis set parameters, and more.

        Usage Examples
        --------------
        **Creating a Custom Abinit File:**

        .. code-block:: python

            from symmstate.abinit.abinit_file import AbinitFile
            from symmstate.slurm_file import SlurmFile
            
            # Assume you have a SlurmFile instance (e.g., with a custom header).
            slurm_obj = SlurmFile(sbatch_header_source="#!/bin/bash\n#SBATCH --time=24:00:00", num_processors=16)
            abinit_file = AbinitFile(abi_file="input.abi", slurm_obj=slurm_obj)
            abinit_file.write_custom_abifile(output_file="custom_input", content="Header content...", coords_are_cartesian=True)

        **Running an Energy Calculation:**

        .. code-block:: python

            abinit_file.run_energy_calculation(host_spec="mpirun -np 16")

        For further examples and integration with the rest of the SymmState CLI, refer to the main documentation.
        """

        # Extended documentation for the SlurmFile class.
        self.SLURM_FILE_DOC =  textwrap.dedent(r"""
        SlurmFile Class Documentation
        ================================

        Overview
        --------
        The **SlurmFile** class manages the creation and execution of SLURM batch scripts with enhanced job monitoring.
        It is derived from the `SymmStateCore` class and is used to configure and submit jobs to a SLURM queue.

        Key Features:
        - **Initialization with Custom Header:**  
            Accepts a multiline string or file path containing the SLURM header (e.g. "#!/bin/bash\n#SBATCH ...")
            along with a specified number of MPI processors.
        
        - **Batch Script Generation:**  
            Generates a batch script using a customizable MPI command template. This script can also include extra shell commands.
        
        - **Job Monitoring:**  
            Provides robust methods to check if all submitted jobs have finished (`all_jobs_finished()`)
            and to wait for job completion (`wait_for_jobs_to_finish()`).

        Initialization
        --------------
        The constructor (`__init__`) accepts:
        - `sbatch_header_source` (str or os.PathLike): The SLURM header text or the path to a file containing it.
        - `num_processors` (int, default=8): The default number of MPI processors for job submission.
        
        Example:
            slurm = SlurmFile(sbatch_header_source="#!/bin/bash\n#SBATCH --time=24:00:00", num_processors=16)
            
        Public Methods
        --------------
        - **write_batch_script(input_file, log_file, batch_name, mpi_command_template, extra_commands):**
        - Writes a SLURM batch script based on a provided MPI command template.
        - Returns the filename of the created batch script.
        
        - **all_jobs_finished():**
        - Checks the status of submitted jobs (using `sacct` or `squeue` as fallbacks)
        - Returns `True` if all tracked jobs are completed, else `False`.
        
        - **wait_for_jobs_to_finish(check_time, check_once):**
        - Enters a waiting loop (or a single-check mode) to monitor job completion.
        - The polling interval is defined by `check_time` in seconds.

        Usage Example in CLI
        ----------------------
        To view this extended documentation through your CLI, add an `examples` command with the option `--slurm-file`:

            symmstate examples --slurm-file

        This will print the above documentation and usage examples for the SlurmFile class.

        """)

