"""Warm the ML-4 reader cache (.cache/reader): pipeline.detect with the reader on, one photo at a
time, one line per photo (seconds, status, cached, images, decisions, winner, margin, note); the
last line is WARMDONE. One process, one llama-server for the whole run (the pipeline starts it on
the first cold read and stops it at exit); run nothing else heavy meanwhile. Afterwards every
ROSETTE_READER=cached run (tools/score.py, tools/judge_mine.py, the comparison pages) reads the
cache and costs only the judge's seconds.

Usage: .venv/Scripts/python.exe tools/reader_warm.py [--ids ID ...]   (default: the 76 photos)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # run as a script
    sys.path.insert(0, str(ROOT))
os.environ["ROSETTE_READER"] = "on"  # the point of the tool: a cold photo asks the model

from rosette.pipeline import detect  # noqa: E402

PHOTOS = ROOT / "reference" / "photos"


def main(argv: list[str]) -> int:
    ids = argv[argv.index("--ids") + 1:] if "--ids" in argv else None
    photos = [PHOTOS / f"{i}.png" for i in ids] if ids else sorted(PHOTOS.glob("*.png"))
    for i, f in enumerate(photos, 1):
        t = time.time()
        try:
            r = detect(f.read_bytes()).reader or {}
            print(f"{i:2d}/{len(photos)} {f.stem} {time.time() - t:.0f}s status={r.get('status')} "
                  f"cached={r.get('cached')} images={r.get('images')} decisions={r.get('decisions')} "
                  f"winner={r.get('winner')} margin={r.get('margin')} note={r.get('note')}", flush=True)
        except Exception as e:  # noqa: BLE001 - one photo's failure never stops the warm-up
            print(f"{i:2d}/{len(photos)} {f.stem} FAILED {type(e).__name__}: {e}", flush=True)
    print("WARMDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
