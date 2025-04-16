# # my_package/main.py
# from my_package.abinit.abinit_workflow import AbinitWorkflow
# from my_package.vasp.vasp_workflow import VaspWorkflow
# from my_package.hpc.slurm_manager import SlurmManager

# def run_calc(code_choice, structure, params):
#     # slurm_manager might hold HPC config like # nodes, time, etc.
#     slurm_manager = SlurmManager(header_file="slurm_header.sh", num_processors=16)

#     if code_choice == "abinit":
#         workflow = AbinitWorkflow(slurm=slurm_manager, workdir="my_abinit_runs")
#     elif code_choice == "vasp":
#         workflow = VaspWorkflow(slurm=slurm_manager, workdir="my_vasp_runs")
#     else:
#         raise ValueError(f"Unknown code choice: {code_choice}")

#     workflow.generate_input_files(structure, params)
#     workflow.submit_job(job_name="test_run")
#     # Optionally wait for job to finish here, or do other steps
#     results = workflow.parse_results(job_name="test_run")
#     print("Results:", results)

# Example usage:
# run_calc("abinit", some_structure_object, {"ecut": 40, "kpoints": [4,4,4]})
# run_calc("vasp", some_structure_object, {"encut": 500, "kspacing": 0.4})
