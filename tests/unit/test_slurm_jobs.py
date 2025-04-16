import unittest
from unittest.mock import patch, MagicMock
import subprocess
import os
from symmstate.slurm import SlurmFile, SlurmHeader


class TestSlurmFile(unittest.TestCase):
    def setUp(self):
        # Create a SlurmHeader for testing purposes
        self.test_header = SlurmHeader(
            job_name="TestJob",
            partition="debug",
            ntasks=1,
            time="01:00:00",
            output="slurm-%j.out",
            additional_lines=["#SBATCH --mem=2G"],
        )
        # Create the SlurmFile instance with 1 processor.
        self.slurm = SlurmFile(
            slurm_header=self.test_header,
            num_processors=1,
            mpi_command_template="mpirun -np {num_procs} dummy < {input_file} > {log}",
        )
        # Set a fake running_jobs list for status checks
        self.slurm.running_jobs = [1234, 5678]

    @patch("subprocess.run")
    def test_all_jobs_finished_when_running(self, mock_run):
        """
        If 'sacct' shows a job in RUNNING state, all_jobs_finished() should return False.
        """
        # Mock sacct output for a RUNNING job
        mock_result = MagicMock()
        # The first two lines are typically headers; the third line is the job state
        mock_result.stdout = "State\nSomeHeader\nRUNNING\n"
        mock_run.return_value = mock_result

        result = self.slurm.all_jobs_finished()
        self.assertFalse(
            result,
            "Expected all_jobs_finished() to return False when a job is running.",
        )

    @patch("subprocess.run")
    def test_all_jobs_finished_when_all_complete(self, mock_run):
        """
        If 'sacct' shows COMPLETED for all jobs, all_jobs_finished() should return True.
        """
        # Mock sacct output for completed jobs
        mock_result = MagicMock()
        mock_result.stdout = "State\nSomeHeader\nCOMPLETED\nCOMPLETED\n"
        mock_run.return_value = mock_result

        result = self.slurm.all_jobs_finished()
        self.assertTrue(
            result,
            "Expected all_jobs_finished() to return True when all jobs are complete.",
        )

    @patch("subprocess.run")
    def test_wait_for_jobs_to_finish_interrupt(self, mock_run):
        """
        Simulate a KeyboardInterrupt during wait_for_jobs_to_finish().
        The method should catch the interrupt without crashing the test.
        """
        mock_result = MagicMock()
        # Always show RUNNING to force loop
        mock_result.stdout = "State\nSomeHeader\nRUNNING\n"
        mock_run.return_value = mock_result

        with patch("time.sleep", side_effect=KeyboardInterrupt):
            try:
                self.slurm.wait_for_jobs_to_finish(check_time=1)
            except KeyboardInterrupt:
                self.fail(
                    "wait_for_jobs_to_finish did not handle KeyboardInterrupt properly."
                )

    def test_write_batch_script(self):
        """
        Verify that the batch script is written with the expected MPI command
        and that it contains Slurm header lines from SlurmFile.
        """
        test_input_file = "dummy_input.in"
        test_log_file = "dummy.log"
        test_batch_name = "dummy_job.sh"

        # Create a dummy input file to pass to write_batch_script
        with open(test_input_file, "w") as f:
            f.write("dummy content")

        # The SlurmFile's mpi_command_template is already set in setUp
        script_path = self.slurm.write_batch_script(
            input_file=test_input_file,
            log_file=test_log_file,
            batch_name=test_batch_name,
        )

        # Check that the script was created
        self.assertTrue(os.path.exists(script_path))
        with open(script_path, "r") as f:
            content = f.read()
            # Check for our MPI command
            self.assertIn("mpirun -np 1 dummy < dummy_input.in > dummy.log", content)
            # Check for at least one Slurm directive from the header (e.g. job-name, partition, etc.)
            self.assertIn("#SBATCH --job-name=TestJob", content)
            self.assertIn("#SBATCH --partition=debug", content)

        # Clean up files
        os.remove(test_input_file)
        os.remove(script_path)


if __name__ == "__main__":
    unittest.main()


# # main.py
# from slurm_header import SlurmHeader
# from slurm_file import SlurmFile

# def main():
#     # 1. Create a SlurmHeader object with your desired directives
#     header = SlurmHeader(
#         job_name="TestJob",
#         partition="debug",
#         ntasks=4,
#         time="01:00:00",
#         output="slurm-%j.out",
#         additional_lines=["#SBATCH --mem=2G"]
#     )

#     # 2. Create the SlurmFile object, passing your SlurmHeader
#     slurm_manager = SlurmFile(
#         slurm_header=header,
#         num_processors=4,
#         mpi_command_template="mpirun -np {num_procs} my_mpi_prog < {input_file} > {log}"
#     )

#     # 3. Write the batch script
#     script_path = slurm_manager.write_batch_script(
#         input_file="test.in",
#         log_file="test.log",
#         batch_name="test_job.sh",
#         extra_commands="echo 'Job is done!'"
#     )

#     # 4. Submit the job
#     job_id = slurm_manager.submit_job(script_path)
#     if not job_id:
#         print("Failed to submit job.")
#         return

#     # 5. Monitor until completion
#     slurm_manager.wait_for_jobs_to_finish(check_time=30)
#     print("All done!")

# if __name__ == "__main__":
#     main()
