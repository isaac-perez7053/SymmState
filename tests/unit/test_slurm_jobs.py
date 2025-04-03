import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
from symmstate.slurm_file import SlurmFile

class TestSlurmFile(unittest.TestCase):
    def setUp(self):
        """Create a temporary file to simulate SLURM batch script."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write(b"#!/bin/bash\n#SBATCH --job-name=test_job\n")
        self.temp_file.close()
        self.slurm = SlurmFile(self.temp_file.name, num_processors=4)

    def tearDown(self):
        """Clean up the temporary file."""
        os.unlink(self.temp_file.name)

    def test_initialization_from_file(self):
        """Test initialization from a file."""
        self.assertTrue("#SBATCH --job-name=test_job" in self.slurm.batch_header)
        self.assertEqual(self.slurm.num_processors, 4)

    def test_initialization_from_string(self):
        """Test initialization from a string header."""
        header_string = "#!/bin/bash\n#SBATCH --job-name=inline_test\n"
        slurm = SlurmFile(header_string, num_processors=8)
        self.assertIn("#SBATCH --job-name=inline_test", slurm.batch_header)
        self.assertEqual(slurm.num_processors, 8)

    @patch("subprocess.run")
    def test_all_jobs_finished(self, mock_run):
        """Test job completion check with mock subprocess responses."""
        self.slurm.running_jobs = [1234, 5678]
        
        # Mock sacct response for completed jobs
        mock_run.return_value.stdout = "State\nCOMPLETED\nCOMPLETED\n"
        
        self.assertTrue(self.slurm.all_jobs_finished())
        self.assertEqual(self.slurm.running_jobs, [])

    @patch("subprocess.run")
    def test_all_jobs_not_finished(self, mock_run):
        """Test scenario where jobs are still running."""
        self.slurm.running_jobs = [1234, 5678]
        
        # Simulate a case where one job is still running
        mock_run.return_value.stdout = "State\nRUNNING\nCOMPLETED\n"

        # Ensure all_jobs_finished() returns False when at least one job is running
        result = self.slurm.all_jobs_finished()
        self.assertFalse(result, "Expected False, but got True")
        self.assertIn(1234, self.slurm.running_jobs, "Job 1234 should still be in the running jobs list")


    @patch("subprocess.run")
    def test_wait_for_jobs_to_finish(self, mock_run):
        """Test waiting for jobs with a single check."""
        self.slurm.running_jobs = [1234]
        mock_run.return_value.stdout = "State\nCOMPLETED\n"
        
        with patch("time.sleep", return_value=None):
            self.slurm.wait_for_jobs_to_finish(check_time=1, check_once=True)
            self.assertEqual(self.slurm.running_jobs, [])

if __name__ == "__main__":
    unittest.main()
