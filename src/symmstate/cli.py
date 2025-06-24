import subprocess
from pathlib import Path
import click
from symmstate.config.symm_state_settings import settings
from symmstate.slurm import SlurmFile
from symmstate.flpz.energy.energy_program import EnergyProgram
from symmstate.flpz.electrotensor.electro_tensor_program import ElectroTensorProgram
from symmstate.flpz.data_analysis import (
    load_flexo_data,
    plot_energy,
    plot_flexo_components,
    plot_flexo_grid,
    plot_varying_components,
)
from symmstate.abinit.abinit_file import AbinitFile
import click
import subprocess
from symmstate.utils import DataParser, Documentation
from symmstate.abinit import AbinitParser


# Define the run_smodes function directly in this file to avoid circular imports.
def run_smodes(smodes_input):
    # Local import to avoid circular dependencies.
    if not Path(settings.SMODES_PATH).is_file():
        raise FileNotFoundError(
            f"SMODES executable not found at: {settings.SMODES_PATH}"
        )
    command = f"{settings.SMODES_PATH} < {smodes_input} "
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"SMODES execution failed: {process.stderr}")
    return process.stdout


BANNER = r"""
+----------------------------------------------------------+
|  ___                  ___ _        _                     |
| / __|_  _ _ __  _ __ / __| |_ __ _| |_ ___               |
| \__ \ || | '  \| '  \\__ \  _/ _` |  _/ -_)              |
| |___/\_, |_|_|_|_|_|_|___/\__\__,_|\__\___|              |
|      |__/                                                |
|                                                          |
| Applications of symmetry in solid state physics          |
+----------------------------------------------------------+
"""


class CustomGroup(click.Group):
    def get_help(self, ctx):
        base_help = super().get_help(ctx)
        return BANNER + "\n\n" + "\n\n" + base_help


@click.group(cls=CustomGroup, invoke_without_command=True)
@click.pass_context
def cli(ctx):
    # Print help (which now includes the banner and additional line) if no command was invoked.
    if ctx.invoked_subcommand is None:
        click.echo(cli.get_help(ctx))


@cli.command()
@click.option(
    "-a",
    "--add",
    multiple=True,
    type=click.Path(),
    help="Add one or more pseudopotential file paths",
)
@click.option(
    "-d",
    "--delete",
    multiple=True,
    type=click.Path(),
    help="Delete one or more pseudopotential file paths",
)
@click.option(
    "-l", "--list", "list_pseudos", is_flag=True, help="List current pseudopotentials"
)
def pseudos(add, delete, list_pseudos):
    """Manage pseudopotential folder paths"""
    from symmstate.pseudopotentials.pseudopotential_manager import (
        PseudopotentialManager,
    )

    if (add or delete) and list_pseudos:
        click.echo(
            "Error: Specify only one action at a time (either add, delete, or list)."
        )
        return

    pm = PseudopotentialManager()
    if add:
        for path in add:
            pm.add_pseudopotential(path)
        click.echo("Pseudopotentials added.")
    elif delete:
        for path in delete:
            pm.delete_pseudopotential(path)
        click.echo("Pseudopotentials deleted.")
    elif list_pseudos:
        if pm.pseudo_registry:
            click.echo("Current pseudopotentials:")
            for name, full_path in pm.pseudo_registry.items():
                click.echo(f"{name} -> {full_path}")
        else:
            click.echo("It's looking empty in here! No pseudopotentials found.")
    else:
        click.echo("Error: No action specified. Use --add, --delete, or --list.")


@cli.command()
@click.option(
    "-l", "--list", is_flag=True, help="List all currently extracted variables"
)
@click.option("--abinit", is_flag=True, help="Extract variables from an Abinit file")
@click.option(
    "--peek",
    "-p",
    type=str,
    help="Comma-separated list of variables to extract (e.g., --peek ecut,kptrlatt,acell)",
)
def variables(list, abinit, peek):
    """List of currently extracted variables"""
    if abinit is False:
        click.echo("Error: Please specify which variables you want to list")

    if list:
        if abinit:

            # List all Abinit variables
            click.echo("All avaiable Abinit variables: \n")

            # Use in build representation function to print all variables
            click.echo(AbinitParser())

    if peek:
        peek_vars = [v.strip() for v in peek.split(",")]
        for var in peek_vars:
            click.echo(
                f"{var}: {AbinitParser.is_supported(var)} - "
                f"{AbinitParser.abinit_variable_descriptions().get(var, 'No description available')}"
            )


@cli.command()
@click.option("--pp-dir", type=click.Path(), help="Set the pseudopotential directory")
@click.option("--working-dir", type=click.Path(), help="Set the working directory")
@click.option("--ecut", type=int, help="Set default energy cutoff (Ha)")
@click.option("--symm-prec", type=float, help="Set symmetry precision")
@click.option("--project-root", type=float, help="Set project root directory")
@click.option(
    "--test-dir",
    type=click.Path(),
    help="Set test directory (relative to the WORKING_DIR)",
)
@click.option(
    "--smodes-path",
    type=click.Path(),
    help="Set the path to the SMODES executable file",
)

@click.option("--templates-dir", type=click.Path(), help="Set the templates directory")

@click.option(
    "--reset-pp-dir",
    is_flag=True,
    help="Reset the pseudopotentials directory to its default value",
)

@click.option("-l", "--list", is_flag=True, help="List current global settings")

def config(pp_dir, working_dir, ecut, symm_prec, project_root, test_dir, smodes_path, templates_dir ,reset_pp_dir, list):
    """Manage global settings of the package"""
    updated = False
    if pp_dir:
        settings.PP_DIR = Path(pp_dir)
        click.echo(f"PP_DIR set to: {settings.PP_DIR}")
    if working_dir:
        settings.WORKING_DIR = Path(working_dir)
        click.echo(f"WORKING_DIR set to: {settings.WORKING_DIR}")
    if ecut:
        settings.DEFAULT_ECUT = ecut
        click.echo(f"DEFAULT_ECUT set to: {settings.DEFAULT_ECUT}")
    if symm_prec:
        settings.SYMM_PREC = symm_prec
        click.echo(f"SYMM_PREC set to: {settings.SYMM_PREC}")
    if project_root:
        settings.PROJECT_ROOT = project_root
        click.echo(f"PROJECT_ROOT set to: {settings.PROJECT_ROOT}")
    if test_dir:
        settings.TEST_DIR = (settings.WORKING_DIR / Path(test_dir)).resolve()
        click.echo(f"TEST_DIR set to: {settings.TEST_DIR}")
    if smodes_path:
        settings.SMODES_PATH = settings.WORKING_DIR / Path(smodes_path).resolve()
        click.echo(f"SMODES_PATH set to: {settings.SMODES_PATH}")

    if reset_pp_dir:
        settings.reset_pp_dir_to_default()
        click.echo(f"PP_DIR reset to default: {settings.PP_DIR}")

    if templates_dir:
        settings.TEMPLATES_DIR = Path(templates_dir).resolve()
        click.echo(f"TEMPLATES_DIR set to: {settings.TEMPLATES_DIR}")
    
    if list:
        click.echo("Current global settings:")
        click.echo(f"PP_DIR: {settings.PP_DIR}")
        click.echo(f"TEMPLATES_DIR: {settings.TEMPLATES_DIR}")
        click.echo(f"SMODES_PATH: {settings.SMODES_PATH}")
        click.echo(f"WORKING_DIR: {settings.WORKING_DIR}")
        click.echo(f"PROJECT_ROOT: {settings.PROJECT_ROOT}")
        click.echo(f"DEFAULT_ECUT: {settings.DEFAULT_ECUT}")
        click.echo(f"SYMM_PREC: {settings.SYMM_PREC}")
        click.echo(f"TEST_DIR: {settings.TEST_DIR}")
    else:
        click.echo("No settings were updated.")


@cli.command()
@click.option(
    "-a", "--add", multiple=True, type=click.Path(), help="Add a template file path"
)
@click.option(
    "-d",
    "--delete",
    multiple=True,
    type=click.Path(),
    help="Delete a template file path",
)
@click.option("-l", "--list", is_flag=True, help="List current templates")
@click.option("--special_list", is_flag=True, help="List special templates only")
@click.option("--is_special", type=click.Path(), help="denote as a special template")
@click.option(
    "-r",
    "--role",
    type=str,
    help="Role of the special template (e.g., 'piezo', 'flexo')",
)
@click.option(
    "--delete_special", is_flag=True, help="Delete a special template by role"
)
@click.option("--unload", type=click.Path(), help="Unload a template file")
def templates(
    add, delete, list, special_list, is_special, role, delete_special, unload
):
    """Manage templates"""
    from symmstate.templates.template_manager import TemplateManager

    if add and delete:
        click.echo("Error: Specify only one action at a time (either add or delete).")
        return

    tm = TemplateManager()
    if add:
        for path in add:
            tm.add_template(path)
        click.echo("Templates added.")
    elif delete:
        for path in delete:
            tm.remove_template(path)
        click.echo("Templates deleted.")
    elif list:
        if tm.template_registry:
            click.echo("Current templates:")
            for name, full_path in tm.template_registry.items():
                click.echo(f"{name} -> {full_path}")
        else:
            click.echo("It's looking empty in here! No templates found.")
    elif special_list:
        if tm.special_templates:
            click.echo("Special templates:")
            for name, full_path in tm.special_templates.items():
                click.echo(f"{name} -> {full_path}")
        else:
            click.echo("It's looking empty in here! No templates found.")
    elif is_special:
        if role:
            if tm.is_special_template(is_special):
                click.echo(f"{is_special} is already special template.")
            else:
                tm.set_special_template(role, is_special)
                click.echo(
                    f"{is_special} has been set as a special template for the role of '{role}'."
                )
        else:
            click.echo(
                "Error: Please specify a role for the special template using --role."
            )
    elif delete_special:
        if role:
            tm.delete_special_template(role)
            click.echo(f"Special template for role '{role}' has been deleted.")
        else:
            click.echo(
                "Error: Please specify a role to delete the special template using --role."
            )
    elif unload:
        if tm.template_exists(unload):
            content = tm.unload_template(unload)
            click.echo(f"Contents of {unload}:\n\n{content}\n")
        else:
            click.echo(f"Template '{unload}' not found.")
    else:
        click.echo(
            "Error: No action specified. Use --add, --delete, --list, or --special_list."
        )


@cli.command()
@click.option("--name", default="EnergyProgram", help="Name of the energy program")
@click.option(
    "--num-datapoints",
    type=int,
    default=3,
    help="Number of perturbed cells to generate",
)
@click.option(
    "--abi-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the Abinit file",
)
@click.option("--min-amp", type=float, default=0.0, help="Minimum amplitude (bohr)")
@click.option("--max-amp", type=float, default=0.5, help="Maximum amplitude (bohr)")
@click.option(
    "--smodes-input",
    type=click.Path(exists=True),
    required=True,
    help="Path to the SMODES input file",
)
@click.option(
    "--target-irrep", type=str, required=True, help="Target irreducible representation"
)
@click.option(
    "--unstable-threshold", type=float, default=-20, help="Unstable threshold value"
)
@click.option("--disp-mag", type=float, default=0.001, help="Displacement magnitude")
def energy(
    name,
    num_datapoints,
    abi_file,
    min_amp,
    max_amp,
    smodes_input,
    target_irrep,
    unstable_threshold,
    disp_mag,
):
    """
    Run the Energy Program.

    Required inputs:
      - abi-file: Path to a valid Abinit file.
      - smodes-input: Path to the SMODES input file.
      - target-irrep: Target irreducible representation.

    Other parameters (with defaults) can be adjusted via options.
    """
    click.echo("Running Energy Program with the following parameters:")
    click.echo(f"Name: {name}")
    click.echo(f"Number of datapoints: {num_datapoints}")
    click.echo(f"Abinit file: {abi_file}")
    click.echo(f"SMODES input: {smodes_input}")
    click.echo(f"Target irreducible representation: {target_irrep}")
    click.echo(f"Min amplitude: {min_amp}")
    click.echo(f"Max amplitude: {max_amp}")
    click.echo(f"Displacement magnitude: {disp_mag}")
    click.echo(f"Unstable threshold: {unstable_threshold}")

    # Create a SlurmFile object using the SLURM header from Settings.
    slurm_header = "".join(
        f"#SBATCH --{key}={value}\n" for key, value in settings.SLURM_HEADER.items()
    )
    slurm_obj = SlurmFile(sbatch_header_source=slurm_header, num_processors=1)

    energy_prog = EnergyProgram(
        name=name,
        num_datapoints=num_datapoints,
        abi_file=abi_file,
        min_amp=min_amp,
        max_amp=max_amp,
        smodes_input=smodes_input,
        target_irrep=target_irrep,
        slurm_obj=slurm_obj,
        symm_prec=settings.SYMM_PREC,
        disp_mag=disp_mag,
        unstable_threshold=unstable_threshold,
    )
    energy_prog.run_program()


@cli.command()
@click.option(
    "--name", default="ElectroTensorProgram", help="Name of the electrotensor program"
)
@click.option(
    "--num-datapoints",
    type=int,
    default=3,
    help="Number of perturbed cells to generate",
)
@click.option(
    "--abi-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the Abinit file",
)
@click.option("--min-amp", type=float, default=0.0, help="Minimum amplitude (bohr)")
@click.option("--max-amp", type=float, default=0.5, help="Maximum amplitude (bohr)")
@click.option(
    "--smodes-input",
    type=click.Path(exists=True),
    required=True,
    help="Path to the SMODES input file",
)
@click.option(
    "--target-irrep", type=str, required=True, help="Target irreducible representation"
)
@click.option(
    "--unstable-threshold", type=float, default=-20, help="Unstable threshold value"
)
@click.option("--disp-mag", type=float, default=0.001, help="Displacement magnitude")
@click.option(
    "--piezo",
    is_flag=True,
    help="Run piezoelectric calculations instead of flexoelectric",
)
def electrotensor(
    name,
    num_datapoints,
    abi_file,
    min_amp,
    max_amp,
    smodes_input,
    target_irrep,
    unstable_threshold,
    disp_mag,
    piezo,
):
    """
    Run the ElectroTensor Program.

    Required inputs:
      - abi-file: Path to a valid Abinit file.
      - smodes-input: Path to the SMODES input file.
      - target-irrep: Target irreducible representation.

    Other parameters (with defaults) can be adjusted via options.
    """
    click.echo("Running ElectroTensor Program with the following parameters:")
    click.echo(f"Name: {name}")
    click.echo(f"Number of datapoints: {num_datapoints}")
    click.echo(f"Abinit file: {abi_file}")
    click.echo(f"SMODES input: {smodes_input}")
    click.echo(f"Target irreducible representation: {target_irrep}")
    click.echo(f"Min amplitude: {min_amp}")
    click.echo(f"Max amplitude: {max_amp}")
    click.echo(f"Displacement magnitude: {disp_mag}")
    click.echo(f"Unstable threshold: {unstable_threshold}")
    click.echo(f"Piezo calculation: {piezo}")

    slurm_header = "".join(
        f"#SBATCH --{key}={value}\n" for key, value in settings.SLURM_HEADER.items()
    )
    slurm_obj = SlurmFile(sbatch_header_source=slurm_header, num_processors=1)

    et_prog = ElectroTensorProgram(
        name=name,
        num_datapoints=num_datapoints,
        abi_file=abi_file,
        min_amp=min_amp,
        max_amp=max_amp,
        smodes_input=smodes_input,
        target_irrep=target_irrep,
        slurm_obj=slurm_obj,
        symm_prec=settings.SYMM_PREC,
        disp_mag=disp_mag,
        unstable_threshold=unstable_threshold,
        piezo_calculation=piezo,
    )
    et_prog.run_program()


@cli.command()
@click.option(
    "--run",
    type=click.Path(exists=True),
    required=False,
    help="Path to the SMODES input file",
)
@click.option(
    "--sym_basis",
    is_flag=True,
    required=False,
    help="Generate the symmetry adapted basis for an Abinit File",
)
@click.argument("abi_file", type=click.Path(exists=True), required=False)
@click.argument("smodes_input", type=click.Path(exists=True), required=False)
@click.argument("irrep", type=str, required=False)
def smodes(run, sym_basis, abi_file, smodes_input, irrep):
    """
    Run SMODES

    This command uses the global SMODES path from Settings.
    """
    if run:
        click.echo("Running SMODES...")
        try:
            result = run_smodes(run)
            click.echo("SMODES output:")
            click.echo(result)
        except Exception as e:
            click.echo(f"Error running SMODES: {e}")

    if sym_basis:
        if not abi_file or not smodes_input or not irrep:
            click.echo("Error: Missing required arguments for --symm_adapted_basis.")
            click.echo("You must specify 'abi_file', 'smodes_input', and 'irrep'.")
            return

        click.echo(
            f"Generating symmetry adapted basis for '{abi_file}' with '{smodes_input}' and irrep '{irrep}'."
        )
        abi_file = AbinitFile(
            abi_file=abi_file, smodes_input=smodes_input, target_irrep=irrep
        )
        click.echo(abi_file)


@cli.group()
def test():
    """Run test suites for individual modules"""
    pass


@test.command()
def abinit_file():
    """Run tests for test_abinit_file.py"""
    test_path = settings.TEST_DIR / "unit" / "test_abinit_file.py"
    subprocess.run(["pytest", str(test_path)], check=True)


@test.command()
def abinit_unit_cell():
    """Run tests for test_abinit_unit_cell.py"""
    test_path = settings.TEST_DIR / "unit" / "test_abinit_unit_cell.py"
    subprocess.run(["pytest", str(test_path)], check=True)


@test.command()
def electrotensor():
    """Run tests for test_electro_tensor.py"""
    test_path = settings.TEST_DIR / "unit" / "test_electro_tensor.py"
    subprocess.run(["pytest", str(test_path)], check=True)


@test.command()
def energy_program():
    """Run tests for test_energy_program.py"""
    test_path = settings.TEST_DIR / "unit" / "test_energy_program.py"
    subprocess.run(["pytest", str(test_path)], check=True)


@test.command()
def perturbations():
    """Run tests for test_perturbations.py"""
    test_path = settings.TEST_DIR / "unit" / "test_perturbations.py"
    subprocess.run(["pytest", str(test_path)], check=True)


@test.command()
def pseudopotential():
    """Run tests for test_pseudopotential.py"""
    test_path = settings.TEST_DIR / "unit" / "test_pseudopotential.py"
    subprocess.run(["pytest", str(test_path)], check=True)


@test.command()
def slurm_jobs():
    """Run tests for test_slurm_jobs.py"""
    test_path = settings.TEST_DIR / "unit" / "test_slurm_jobs.py"
    subprocess.run(["pytest", str(test_path)], check=True)


@test.command()
def smodes_calculator():
    """Run tests for test_smodes_calculator.py"""
    test_path = settings.TEST_DIR / "unit" / "test_smodes_calculator.py"
    subprocess.run(["pytest", str(test_path)], check=True)


@test.command()
def template_manager():
    """Run tests for test_template_manager.py"""
    test_path = settings.TEST_DIR / "unit" / "test_template_manager.py"
    subprocess.run(["pytest", str(test_path)], check=True)


@test.command()
def unit_cell_module():
    """Run tests for test_unit_cell_module.py"""
    test_path = settings.TEST_DIR / "unit" / "test_unit_cell_module.py"
    subprocess.run(["pytest", str(test_path)], check=True)


@test.command()
def test_all():
    """Run all tests at once using pytest"""
    test_path = settings.TEST_DIR / "unit"
    if not test_path.exists():
        click.echo(f"Test directory not found: {test_path}")
        return
    subprocess.run(["pytest", str(test_path)], check=True)


@cli.command()
@click.option(
    "--results-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the results file",
)
@click.option(
    "--analysis-type",
    type=click.Choice(["energy", "flexo", "grid", "varying"]),
    required=True,
    help="Type of data analysis to perform",
)
@click.option("--save", is_flag=True, help="Save the generated plot to a file")
@click.option("--filename", default="analysis_plot", help="Filename for the saved plot")
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Threshold value for 'varying' analysis (optional)",
)
def data_analysis(results_file, analysis_type, save, filename, threshold):
    """Perform data analysis on a results file"""
    amplitudes, energies, flexo_amps, flexo_tensors = load_flexo_data(results_file)

    if analysis_type == "energy":
        ax = plot_energy(amplitudes, energies)
    elif analysis_type == "flexo":
        ax = plot_flexo_components(flexo_amps, flexo_tensors)
    elif analysis_type == "grid":
        fig = plot_flexo_grid(flexo_amps, flexo_tensors)
    elif analysis_type == "varying":
        fig = plot_varying_components(flexo_amps, flexo_tensors, threshold=threshold)

    if save:
        if analysis_type in ["flexo", "grid", "varying"]:
            if analysis_type == "grid":
                fig.savefig(f"{filename}_grid.png", bbox_inches="tight")
                click.echo(f"Grid plot saved as {filename}_grid.png")
            elif analysis_type == "varying":
                fig.savefig(f"{filename}_varying.png", bbox_inches="tight")
                click.echo(f"Varying components plot saved as {filename}_varying.png")
            else:
                ax.get_figure().savefig(f"{filename}_flexo.png", bbox_inches="tight")
                click.echo(f"Flexoelectric plot saved as {filename}_flexo.png")
        else:
            ax.get_figure().savefig(f"{filename}_energy.png", bbox_inches="tight")
            click.echo(f"Energy plot saved as {filename}_energy.png")
    else:
        if analysis_type in ["flexo", "grid", "varying"]:
            fig.show()
        else:
            ax.get_figure().show()


@cli.command()
@click.option(
    "--results-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the results file produced",
)
@click.option(
    "--analysis-type",
    type=click.Choice(["energy", "flexo", "grid", "varying"]),
    required=True,
    help="Type of data analysis to perform",
)
@click.option("--save", is_flag=True, help="Save the generated plot to a file")
@click.option("--filename", default="analysis_plot", help="Filename for the saved plot")
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Threshold value for 'varying' analysis (optional)",
)
def data_analysis(results_file, analysis_type, save, filename, threshold):
    """Perform data analysis on a results file"""
    amplitudes, energies, flexo_amps, flexo_tensors = load_flexo_data(results_file)

    if analysis_type == "energy":
        ax = plot_energy(amplitudes, energies)
    elif analysis_type == "flexo":
        ax = plot_flexo_components(flexo_amps, flexo_tensors)
    elif analysis_type == "grid":
        fig = plot_flexo_grid(flexo_amps, flexo_tensors)
    elif analysis_type == "varying":
        fig = plot_varying_components(flexo_amps, flexo_tensors, threshold=threshold)

    if save:
        if analysis_type in ["flexo", "grid", "varying"]:
            if analysis_type == "grid":
                fig.savefig(f"{filename}_grid.png", bbox_inches="tight")
                click.echo(f"Grid plot saved as {filename}_grid.png")
            elif analysis_type == "varying":
                fig.savefig(f"{filename}_varying.png", bbox_inches="tight")
                click.echo(f"Varying components plot saved as {filename}_varying.png")
            else:
                ax.get_figure().savefig(f"{filename}_flexo.png", bbox_inches="tight")
                click.echo(f"Flexoelectric plot saved as {filename}_flexo.png")
        else:
            ax.get_figure().savefig(f"{filename}_energy.png", bbox_inches="tight")
            click.echo(f"Energy plot saved as {filename}_energy.png")
    else:
        if analysis_type in ["flexo", "grid", "varying"]:
            fig.show()
        else:
            ax.get_figure().show()


@cli.command()
@click.option(
    "--flexotensor",
    type=click.Path(exists=True),
    required=False,
    help="Grab the flexoelectric tensor of an Abinit file",
)
@click.option(
    "--piezotensor",
    type=click.Path(exists=True),
    required=False,
    help="Grab the piezoelectric tensor of an Abinit file",
)
@click.option(
    "--energy",
    type=click.Path(exists=True),
    required=False,
    help="Grab the energy from an Abinit file",
)
def grab(flexotensor, piezotensor, energy):
    """Grab various quantities from an Abinit file"""

    if flexotensor:
        tensor = DataParser.grab_flexo_tensor(flexotensor)
        click.echo(
            f"Flexoelectric tensor located in the file {flexotensor} is:\n" f"{tensor}"
        )

    if piezotensor:
        tensor_clamped, tensor_relaxed = DataParser.grab_flexo_tensor(piezotensor)
        click.echo(
            f"Piezoelectric tensors located in the file {piezotensor} are:\n"
            f"Clamped piezoelectric tensor:\n{tensor_clamped}\n"
            f"Relaxed piezoelectric tensor:\n{tensor_relaxed}"
        )

    if energy:
        energy_value = DataParser.grab_energy(energy)
        click.echo(f"Energy located in the file {energy} is:\n" f"{energy_value}")


# Create a single Documentation instance.
docs = Documentation()

# Define constant strings.
UNIT_CELL_DOC = docs.UNITCELL
ABINIT_UNIT_CELL_DOC = docs.ABINIT_UNIT_CELL_DOC
ABINIT_FILE_DOC = docs.ABINIT_FILE_DOC
SLURM_FILE_DOC = docs.SLURM_FILE_DOC


@cli.command()
@click.option(
    "--unit-cell",
    is_flag=True,
    help="Show extended documentation for the UnitCell class",
)
@click.option(
    "--abinit-cell",
    is_flag=True,
    help="Show extended documentation for the AbinitUnitCell class",
)
@click.option(
    "--abinit-file",
    is_flag=True,
    help="Show extended documentation for the AbinitFile class",
)
@click.option(
    "--slurm-file",
    is_flag=True,
    help="Show extended documentation for the SlurmFile class",
)
def examples(unit_cell, abinit_cell, abinit_file, slurm_file):
    """Displays examples and extended documentation"""
    if unit_cell:
        click.echo(UNIT_CELL_DOC)
    elif abinit_cell:
        click.echo(ABINIT_UNIT_CELL_DOC)
    elif abinit_file:
        click.echo(ABINIT_FILE_DOC)
    elif slurm_file:
        click.echo(SLURM_FILE_DOC)
    else:
        click.echo("No example selected. Please specify one of the available options.")


if __name__ == "__main__":
    cli()
