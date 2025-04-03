import os
from pathlib import Path
import click
from symmstate.config.settings import Settings
from symmstate.pseudopotentials.pseudopotential_manager import PseudopotentialManager
from symmstate.templates.template_manager import TemplateManager

@click.group()
def cli():
    """SymmState: Manage folder interactions"""

@cli.command()
@click.option("-a", "--add", multiple=True, type=click.Path(), help="Add one or more pseudo potential file paths")
@click.option("-d", "--delete", multiple=True, type=click.Path(), help="Delete one or more pseudo potential file paths")
@click.option("-l", "--list", "list_pseudos", is_flag=True, help="List current pseudopotentials")
def pseudos(add, delete, list_pseudos):
    """Manage pseudo potential folder paths"""
    # Ensure only one action is specified.
    if (add or delete) and list_pseudos:
        click.echo("Error: Please specify only one action at a time (either add, delete, or list).")
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
            click.echo("No pseudopotentials found.")
    else:
        click.echo("Error: No action specified. Use --add, --delete, or --list.")

@cli.command()
@click.option("--pp-dir", type=click.Path(), help="Set the pseudopotential directory")
@click.option("--smodes-path", type=click.Path(), help="Set the SMODES executable path")
@click.option("--working-dir", type=click.Path(), help="Set the working directory")
@click.option("--ecut", type=int, help="Set default energy cutoff (hartree)")
@click.option("--symm-prec", type=float, help="Set symmetry precision")
@click.option("--kpt-density", type=float, help="Set default k-point density")
@click.option("--slurm-time", type=str, help="Set SLURM time")
@click.option("--slurm-nodes", type=int, help="Set SLURM nodes")
@click.option("--slurm-ntasks", type=int, help="Set SLURM tasks per node")
@click.option("--slurm-mem", type=str, help="Set SLURM memory")
@click.option("--environment", type=str, help="Set environment")
def config(pp_dir, smodes_path, working_dir, ecut, symm_prec, kpt_density, slurm_time, slurm_nodes, slurm_ntasks, slurm_mem, environment):
    """Manage global settings of the package"""
    updated = False
    if pp_dir:
        Settings.PP_DIR = Path(pp_dir)
        updated = True
    if smodes_path:
        Settings.SMODES_PATH = Path(smodes_path)
        updated = True
    if working_dir:
        Settings.WORKING_DIR = Path(working_dir)
        updated = True
    if ecut:
        Settings.DEFAULT_ECUT = ecut
        updated = True
    if symm_prec:
        Settings.SYMM_PREC = symm_prec
        updated = True
    if kpt_density:
        Settings.DEFAULT_KPT_DENSITY = kpt_density
        updated = True
    if slurm_time:
        Settings.SLURM_HEADER["time"] = slurm_time
        updated = True
    if slurm_nodes:
        Settings.SLURM_HEADER["nodes"] = slurm_nodes
        updated = True
    if slurm_ntasks:
        Settings.SLURM_HEADER["ntasks-per-node"] = slurm_ntasks
        updated = True
    if slurm_mem:
        Settings.SLURM_HEADER["mem"] = slurm_mem
        updated = True
    if environment:
        Settings.ENVIRONMENT = environment
        updated = True

    if updated:
        click.echo("Settings updated:")
        click.echo(f"PP_DIR: {Settings.PP_DIR}")
        click.echo(f"SMODES_PATH: {Settings.SMODES_PATH}")
        click.echo(f"WORKING_DIR: {Settings.WORKING_DIR}")
        click.echo(f"DEFAULT_ECUT: {Settings.DEFAULT_ECUT}")
        click.echo(f"SYMM_PREC: {Settings.SYMM_PREC}")
        click.echo(f"DEFAULT_KPT_DENSITY: {Settings.DEFAULT_KPT_DENSITY}")
        click.echo(f"SLURM_HEADER: {Settings.SLURM_HEADER}")
        click.echo(f"ENVIRONMENT: {Settings.ENVIRONMENT}")
    else:
        click.echo("No settings were updated.")

@cli.command()
@click.option("-a", "--add", multiple=True, type=click.Path(), help="Add a template file path")
@click.option("-d", "--delete", multiple=True, type=click.Path(), help="Delete a template file path")
def templates(add, delete):
    """Manage templates"""
    if add and delete:
        click.echo("Error: Please specify only one action at a time (either add or delete).")
        return

    tm = TemplateManager()
    
    if add:
        for path in add:
            tm.create_template(path, os.path.basename(path))
        click.echo("Templates added.")
    elif delete:
        for path in delete:
            tm.remove_template(os.path.basename(path))
        click.echo("Templates deleted.")
    else:
        click.echo("Error: No action specified. Use --add or --delete.")

if __name__ == "__main__":
    cli()

