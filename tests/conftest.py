# tests/conftest.py
import pytest
from symmstate import UnitCell

@pytest.fixture
def sample_abi_file(tmp_path):
    content = """..."""
    path = tmp_path/"test.abi"
    path.write_text(content)
    return path

@pytest.fixture
def sample_unit_cell(sample_abi_file):
    return UnitCell(abi_file=sample_abi_file)

@pytest.fixture
def mock_slurm(monkeypatch):
    class MockSlurm:
        def __init__(self):
            self.submitted_commands = []
            
        def mock_submit(self, args):
            self.submitted_commands.append(" ".join(args))
            return f"Submitted batch job {len(self.submitted_commands)}"
    
    mock = MockSlurm()
    monkeypatch.setattr(subprocess, "run", mock.mock_submit)
    return mock