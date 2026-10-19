import os

import pytest


@pytest.fixture(autouse=True)
def _clear_slurm_environment(monkeypatch):
    """Hide any enclosing SLURM allocation (e.g., an interactive or tunnel job) from Lightning, which would otherwise detect
    it as a SLURM-managed run and reject or reconfigure the trainer (e.g., "You set `--ntasks=4`..."). Subprocesses started by
    the test inherit the cleared environment, and it is restored afterwards."""
    for name in [name for name in os.environ if name.startswith("SLURM_")]:
        monkeypatch.delenv(name)
