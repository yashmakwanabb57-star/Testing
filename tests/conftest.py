"""SAM 2 pieces are off in the tests unless a test turns them on (ML-2): the suite checks the
rules path, and a photo missing from the pieces cache would run the model for a minute."""
import pytest


@pytest.fixture(autouse=True)
def _pieces_off(monkeypatch):
    monkeypatch.setenv("ROSETTE_SAM2", "off")
