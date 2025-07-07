from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SlurmHeader:
    """
    Container for SLURM '#SBATCH' directive settings.

    Each non-empty field in this dataclass generates a corresponding '#SBATCH' line in a job
    submission script. Use 'additional_lines' to include any custom directives not covered by
    the predefined attributes.

    Attributes:
        job_name (Optional[str]):
            Name of the job (--job-name).
        partition (Optional[str]):
            Partition or queue to submit the job to (--partition).
        ntasks (Optional[int]):
            Total number of MPI tasks (--ntasks).
        cpus_per_task (Optional[int]):
            Number of CPU cores per task (--cpus-per-task).
        time (Optional[str]):
            Maximum runtime in the format 'D-HH:MM:SS' or 'HH:MM:SS' (--time).
        output (Optional[str]):
            Path or pattern for the STDOUT file (--output), e.g., 'slurm-%j.out'.
        error (Optional[str]):
            Path or pattern for the STDERR file (--error), e.g., 'slurm-%j.err'.
        additional_lines (List[str]):
            List of extra '#SBATCH' directive lines, e.g., ['#SBATCH --mem=4G',
            '#SBATCH --mail-type=END'].
    """
    job_name: Optional[str] = None
    partition: Optional[str] = None
    ntasks: Optional[int] = None
    cpus_per_task: Optional[int] = None
    time: Optional[str] = None  # e.g. "24:00:00"
    output: Optional[str] = None  # e.g. "slurm-%j.out"
    error: Optional[str] = None
    additional_lines: List[str] = field(default_factory=list)

    def to_string(self) -> str:
        """
        Generate the '#SBATCH' directive block.

        Returns:
            A string containing each SLURM directive as a separate '#SBATCH' line.
        """
        lines = []
        if self.job_name:
            lines.append(f"#SBATCH --job-name={self.job_name}")
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.ntasks:
            lines.append(f"#SBATCH --ntasks={self.ntasks}")
        if self.cpus_per_task:
            lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        if self.time:
            lines.append(f"#SBATCH --time={self.time}")
        if self.output:
            lines.append(f"#SBATCH --output={self.output}")
        if self.error:
            lines.append(f"#SBATCH --error={self.error}")

        # Append any extra #SBATCH lines, e.g. --mem=4G, --mail-type=ALL, etc.
        for line in self.additional_lines:
            lines.append(line)

        return "\n".join(lines)
