import numpy as np
from typing import List, Tuple, Dict
import time

from symmstate.abinit import AbinitFile, MrgddbFile, AnaddbFile
from symmstate.flpz import FlpzCore
from symmstate.utils import DataParser, get_unique_filename
from symmstate.slurm import SlurmFile


class Perturbations(FlpzCore):
    """
    A class that facilitates the generation and analysis of perturbations in
    an Abinit unit cell, enabling the calculation of energy, piezoelectric,
    and flexoelectric properties.
    """

    def __init__(self):
        """
        Initializes the Perturbations instance
        """
        pass

    @staticmethod
    def generate_perturbations(
        num_datapoints: int,
        abinit_file: AbinitFile,
        min_amp: int,
        max_amp: int,
        perturbation: np.ndarray,
    ) -> Tuple[List[float], List[AbinitFile]]:
        """
        Generates perturbed unit cells based on the given number of datapoints.
        Returns:
            list: A list of perturbed AbinitFile objects.
        """

        # Calculate the step size.
        step_size = (max_amp - min_amp) / (num_datapoints - 1)
        perturbed_objects = []
        list_amps = []
        for i in range(num_datapoints):
            current_amp = min_amp + i * step_size
            list_amps.append(current_amp)
            # Compute perturbed values and obtain a new AbinitFile object.
            perturbed_values = current_amp * perturbation
            perturbation_result = abinit_file.perturbation(
                perturbed_values, coords_are_cartesian=True
            )
            perturbed_objects.append(perturbation_result)

        return list_amps, perturbed_objects

    @staticmethod
    def calculate_energy_of_perturbations(
        perturbations: List[AbinitFile],
        slurm_obj: SlurmFile,
    ) -> Tuple[List[str], List[float]]:
        """
        Calculate the energy associated with a list of perturbed AbinitFiles
        """
        list_abo_files = []
        for i, obj in enumerate(perturbations):
            # Now explicitly pass the slurm_obj and output file name
            _ = obj.execute_energy_calculation(
                output_file=f"{obj.filename}_energy_{i}",
                slurm_obj=slurm_obj,
                batch_name=f"{obj.filename}_energy_{i}.sh",
                log_file=f"{obj.filename}_energy_{i}.log",
            )
            list_abo_files.append(f"{obj.filename}_energy_{i}.abo")

        slurm_obj.wait_for_jobs_to_finish(check_time=90)

        # Append energies
        results = []
        for i, obj in enumerate(perturbations):
            results.append(DataParser.grab_energy(f"{obj.filename}_energy_{i}.abo"))

        return list_abo_files, results

    @staticmethod
    def calculate_piezo_of_perturbations(
        perturbations: List[AbinitFile], slurm_obj: SlurmFile, sleep_time: int = 10
    ):
        """
        Runs piezoelectric calculations for each perturbed object and stores the
        energies and piezoelectric tensors (both clamped and relaxed) in self.results.
        """
        list_abo_files = []
        for i, obj in enumerate(perturbations):
            output_file = obj.execute_piezo_calculation(
                output_file=f"{obj.filename}_piezo_{i}",
                slurm_obj=slurm_obj,
                batch_name=f"{obj.filename}_piezo_{i}.sh",
                log_file=f"{obj.filename}_piezo_{i}.log",
            )
            list_abo_files.append(f"{obj.filename}_piezo_{i}.abo")

        slurm_obj.wait_for_jobs_to_finish(check_time=300)

        # Create and execute the mrgddb files
        mrgddb_output_files = []
        for i, obj in enumerate(perturbations):
            output_file = f"{obj.filename}_mrgddb_{i}.ddb"
            ddb_files = [
                f"{obj.filename}_piezo_{i}_gen_output_DS1_DDB",
                f"{obj.filename}_piezo_{i}_gen_output_DS4_DDB",
            ]

            mrgddb_input = MrgddbFile.write(
                input_name=f"{obj.filename}_mrgddb_{i}.in",
                ddb_files=ddb_files,
                output_file=f"{obj.filename}_mrgddb_{i}.ddb",
                description=f"Mrgddb input for perturbation {i} of the file {obj.filename}",
            )

            MrgddbFile.execute(
                input_file=mrgddb_input,
                log_name=f"{obj.filename}_mrgddb.log",
                sleep_time=1,
            )
            mrgddb_output_files.append(output_file)

        # Ensure the mrgddb files are finished running
        time.sleep(sleep_time)

        # Creation and execution of anaddb files
        anaddb_piezo_files = []
        content = """
! Input file for the anaddb code
elaflag 3
piezoflag 3
instrflag 1
""".strip()

        for i, obj in enumerate(perturbations):
            output_file = f"{obj.filename}_anaddb_{i}.out"
            anaddb_files_input = AnaddbFile.write(
                in_content=content,
                anaddb_in=f"{obj.filename}_anaddb_{i}.in",
                out_name=output_file,
                ddb=mrgddb_output_files[i],
                files_name=f"{obj.filename}_anaddb_{i}.files",
            )

            AnaddbFile.execute(
                files_name=anaddb_files_input, log_name=f"{obj.filename}_anaddb_{i}.log"
            )
            anaddb_piezo_files.append(output_file)

        # Ensure Anaddb files are finished runnign
        time.sleep(sleep_time)

        results = {}

        # Clear any previous energy and piezo results.
        results["energies"] = []
        results["clamped_piezotensors"] = []
        results["relaxed_piezotensors"] = []
        for i, obj in enumerate(perturbations):
            energy = DataParser.grab_energy(list_abo_files[i])
            clamped_tensor, relaxed_tensor = DataParser.grab_piezo_tensor(
                anaddb_file=anaddb_piezo_files[i]
            )
            results["energies"].append(energy)
            results["clamped_piezotensors"].append(clamped_tensor)
            results["relaxed_piezotensors"].append(relaxed_tensor)

        return results

    @staticmethod
    def calculate_flexo_of_perturbations(
        perturbations: List[AbinitFile], slurm_obj: SlurmFile, sleep_time: int = 10
    ):
        """
        Runs flexoelectric calculations for each perturbed object and stores the
        energies, flexoelectric tensors, and piezoelectric tensors in self.results.
        """

        list_abo_files = []
        for i, obj in enumerate(perturbations):
            output_file = obj.execute_flexo_calculation(
                output_file=f"{obj.filename}_flexo_{i}",
                slurm_obj=slurm_obj,
                batch_name=f"{obj.filename}_flexo_{i}.sh",
                log_file=f"{obj.filename}_flexo_{i}.log",
            )
            list_abo_files.append(f"{obj.filename}_flexo_{i}.abo")

        # Create and execute the mrgddb files
        mrgddb_output_files = []
        for i, obj in enumerate(perturbations):
            output_file = f"{obj.filename}_mrgddb_{i}.ddb"
            ddb_files = [
                f"{obj.filename}_flexo_{i}_gen_output_DS1_DDB",
                f"{obj.filename}_flexo_{i}_gen_output_DS4_DDB",
                f"{obj.filename}_flexo_{i}_gen_output_DS5_DDB",
            ]

            mrgddb_input = MrgddbFile.write(
                input_name=f"{obj.filename}_mrgddb_{i}.in",
                ddb_files=ddb_files,
                output_file=f"{obj.filename}_mrgddb_{i}.ddb",
                description=f"Mrgddb input for perturbation {i} of the file {obj.filename}",
            )

            MrgddbFile.execute(
                input_file=mrgddb_input,
                log_name=f"{obj.filename}_mrgddb.log",
                sleep_time=1,
            )
            mrgddb_output_files.append(output_file)

        # Ensure the mrgddb files are finished running
        time.sleep(sleep_time)

        # Creation and execution of anaddb files
        anaddb_files = []
        content = """
! Input file for the anaddb code
elaflag 3
piezoflag 3
instrflag 1
flexoflag 1
""".strip()

        for i, obj in enumerate(perturbations):
            output_file = f"{obj.filename}_anaddb_{i}.out"
            anaddb_files_input = AnaddbFile.write(
                in_content=content,
                anaddb_in=f"{obj.filename}_anaddb_{i}.in",
                out_name=output_file,
                ddb=mrgddb_output_files[i],
                files_name=f"{obj.filename}_anaddb_{i}.files",
            )

            AnaddbFile.execute(
                files_name=anaddb_files_input, log_name=f"{obj.filename}_anaddb_{i}.log"
            )
            anaddb_files.append(output_file)

        # Ensure Anaddb files are finished runnign
        time.sleep(sleep_time)

        # Store required information
        results = {}
        results["energies"] = []
        results["clamped_piezotensors"] = []
        results["relaxed_piezotensors"] = []
        results["flexotensors"] = []
        # Grab the desired information from the output files
        for i, obj in enumerate(perturbations):
            energy = DataParser.grab_energy(list_abo_files[i])
            flexotensor = DataParser.grab_flexo_tensor(anaddb_file=anaddb_files[i])
            clamped_tensor, relaxed_tensor = DataParser.grab_piezo_tensor(
                anaddb_file=anaddb_files[i]
            )
            results["energies"].append(energy)
            results["flexotensors"].append(flexotensor)
            results["clamped_piezotensors"].append(clamped_tensor)
            results["relaxed_piezotensors"].append(relaxed_tensor)

        return results

    def record_data(
        abi_file: AbinitFile,
        datafile_name: str,
        results: Dict,
        perturbation: np.ndarray,
        list_amps: List,
    ):
        """
        Writes a summary of the run to a file in the format expected by data-analysis.py.
        This function retrieves the relevant arrays from self.results, including
        flexo- and piezo-related tensors (clamped & relaxed).

        Keys expected in self.results:
        - "amps" -> list of amplitude values
        - "energies" -> list of energies
        - "flexo_amps" -> list of amplitudes used specifically for flexo data
        - "flexo_tensors" -> list of 9x6 arrays for flexo
        - "clamped_piezo_tensors" -> list of NxM arrays for clamped piezo data
        - "relaxed_piezo_tensors" -> list of NxM arrays for relaxed piezo data
        """
        # Useful when analyzing multiple unstable phonons
        data_file = get_unique_filename(datafile_name)
        with open(data_file, "w") as f:
            # Basic info
            f.write("Data File\n")
            f.write("Basic Cell File Name:\n")
            f.write(f"{abi_file.filename}\n\n")
            f.write("Perturbation Associated with Run:\n")
            f.write(f"{perturbation}\n\n")

            # Extract from self.results
            amps = list_amps
            energies = results.get("energies", [])
            flexo_amps = results.get("amps", [])
            flexo_tensors = results.get("flexotensors", [])

            # Piezo data (clamped, relaxed)
            clamped_tensors = results.get("clamped_piezotensors", [])
            relaxed_tensors = results.get("relaxed_piezotensors", [])

            # Required lines for data-analysis.py
            f.write(f"List of Amplitudes: {amps}\n")
            f.write(f"List of Energies: {energies}\n")
            f.write(f"List of Flexo Amplitudes: {flexo_amps}\n")

            # 1) Flexo Tensors
            f.write("List of Flexo Electric Tensors:\n")
            for tensor in flexo_tensors:
                for row in tensor:
                    row_str = " ".join(str(x) for x in row)
                    f.write(f"[ {row_str} ]\n")

            # 2) Clamped Piezo Tensors
            f.write("List of Clamped Piezo Tensors:\n")
            for tensor in clamped_tensors:
                for row in tensor:
                    row_str = " ".join(str(x) for x in row)
                    f.write(f"[ {row_str} ]\n")

            # 3) Relaxed Piezo Tensors
            f.write("List of Relaxed Piezo Tensors:\n")
            for tensor in relaxed_tensors:
                for row in tensor:
                    row_str = " ".join(str(x) for x in row)
                    f.write(f"[ {row_str} ]\n")
