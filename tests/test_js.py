"""Runs the Node tests in tests/js (workbench logic and the shop-coloring port).
Skipped when node is not on PATH; no npm packages are needed."""
import shutil
import subprocess
from pathlib import Path

import pytest

JS_TESTS = Path(__file__).resolve().parent / "js"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not on PATH")
def test_node_suite():
    files = sorted(str(p) for p in JS_TESTS.glob("*.test.mjs"))
    assert files
    r = subprocess.run([NODE, "--test", *files], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
