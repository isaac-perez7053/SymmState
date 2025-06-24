from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SlurmHeader:
    """
    Container for SLURM '#SBATCH' directive settings.

    Each non-empty field in this dataclass generates a corresponding '#SBATCH' line for batch
    script headers. Additional custom directives can be specified via the 'additional_lines' list.
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
