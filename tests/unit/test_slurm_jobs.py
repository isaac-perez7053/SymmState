import unittest
from unittest.mock import patch, MagicMock
import subprocess
from symmstate.slurm_file import SlurmFile
import os

# We'll create a dummy header for testing purposes.
DUMMY_HEADER = "#SBATCH --time=01:00:00\n#SBATCH --nodes=1\n"

class TestSlurmFile(unittest.TestCase):
    def setUp(self):
        # Create a SlurmFile instance with a dummy header and one processor.
        self.slurm = SlurmFile(sbatch_header_source=DUMMY_HEADER, num_processors=1)
        # Set a fake running_jobs list.
        self.slurm.running_jobs = [1234, 5678]

    @patch("subprocess.run")
    def test_all_jobs_finished_when_running(self, mock_run):
        # Simulate output for a job that is running.
        # Here, we simulate that the 'sacct' command returns a header and then a "RUNNING" state.
        mock_result = MagicMock()
        # Return a stdout where the header is the first 2 lines, then a "RUNNING" state.
        mock_result.stdout = "State\nSomeHeader\nRUNNING\n"
        mock_run.return_value = mock_result

        # Since one job is still running, all_jobs_finished() should return False.
        result = self.slurm.all_jobs_finished()
        self.assertFalse(result, "Expected all_jobs_finished() to return False when a job is running.")

    @patch("subprocess.run")
    def test_all_jobs_finished_when_all_complete(self, mock_run):
        # Simulate output for jobs that are all complete.
        mock_result = MagicMock()
        # The output simulates header and then only "COMPLETED" states.
        mock_result.stdout = "State\nSomeHeader\nCOMPLETED\nCOMPLETED\n"
        mock_run.return_value = mock_result

        # With both jobs complete, all_jobs_finished() should return True.
        result = self.slurm.all_jobs_finished()
        self.assertTrue(result, "Expected all_jobs_finished() to return True when all jobs are complete.")

    @patch("subprocess.run")
    def test_wait_for_jobs_to_finish_interrupt(self, mock_run):
        # Simulate that the jobs never finish by always returning "RUNNING".
        mock_result = MagicMock()
        mock_result.stdout = "State\nSomeHeader\nRUNNING\n"
        mock_run.return_value = mock_result

        # We'll simulate a KeyboardInterrupt when wait_for_jobs_to_finish is called.
        with patch("time.sleep", side_effect=KeyboardInterrupt):
            # We expect that wait_for_jobs_to_finish eventually exits on KeyboardInterrupt.
            try:
                self.slurm.wait_for_jobs_to_finish(check_time=1)
            except KeyboardInterrupt:
                self.fail("wait_for_jobs_to_finish did not handle KeyboardInterrupt properly.")

    def test_write_batch_script(self):
        # Write a temporary batch script.
        test_input_file = "dummy_input.in"
        test_log_file = "dummy.log"
        test_batch_name = "dummy_job.sh"
        # Create a dummy input file.
        with open(test_input_file, "w") as f:
            f.write("dummy content")

        script_path = self.slurm.write_batch_script(
            input_file=test_input_file,
            log_file=test_log_file,
            batch_name=test_batch_name,
            mpi_command_template="mpirun -np {num_procs} dummy < {input_file} > {log}"
        )
        self.assertTrue(os.path.exists(script_path))
        with open(script_path, "r") as f:
            content = f.read()
            self.assertIn("mpirun -np 1 dummy < dummy_input.in > dummy.log", content)
        # Clean up the temporary file.
        os.remove(test_input_file)
        os.remove(script_path)

if __name__ == '__main__':
    unittest.main()
