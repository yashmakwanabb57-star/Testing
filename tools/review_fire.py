"""Which of the 76 photos a rule changes (review step 6).

    .venv/Scripts/python.exe tools/review_fire.py _rose_from_small_star
    .venv/Scripts/python.exe tools/review_fire.py --part points-flower

A function name in rosette.pipeline (a rule that returns True when it fires) is wrapped and
counted; --part lists the photos whose final head has a part with that id instead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ROSETTE_READER", "off")
os.environ.setdefault("ROSETTE_SAM2", "off")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rosette.pipeline as pl  # noqa: E402
from rosette.vision.templates import design_key  # noqa: E402


def main(args: list[str]) -> None:
    part = args[1] if args[0] == "--part" else None
    hits: list = []
    if part is None:
        rule = getattr(pl, args[0])

        def wrapped(*a, **k):
            r = rule(*a, **k)
            hits.append(bool(r))
            return r

        setattr(pl, args[0], wrapped)
    for f in sorted(Path("reference/photos").glob("*.png")):
        hits.clear()
        m, _, _ = pl.analyze_with_context(f.read_bytes(), exclude_key=design_key(f.stem))
        if any(hits) or (part and any(q.id == part for q in m.parts)):
            print("fires on", f.stem, flush=True)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:])
