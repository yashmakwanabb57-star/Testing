"""One photo for review: photo | ours | designer side by side, and the head parts.

    .venv/Scripts/python.exe tools/review_show.py S28-15 [out_dir]

Writes <out_dir>/cur.png (comparison) and <out_dir>/z.png (our head, close up), prints the
head parts. out_dir defaults to output/review.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ROSETTE_READER", "off")
os.environ.setdefault("ROSETTE_SAM2", "off")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from rosette.pipeline import analyze_with_context, render  # noqa: E402
from rosette.vision.templates import design_key  # noqa: E402
from tools.rasterize import rasterize  # noqa: E402

KEYS = ("cx", "cy", "r", "r_inner", "r_outer", "kind", "count", "rotation_deg", "template_id")


def main(sid: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    m, _, _ = analyze_with_context(Path(f"reference/photos/{sid}.png").read_bytes(), exclude_key=design_key(sid))
    for q in m.parts:
        if q.role != "tail_stripe":
            print(q.id, q.role, q.color_group, q.z, {k: q.params[k] for k in KEYS if k in q.params})
    svg = render(m).svg
    a = rasterize(svg, 850, None)
    d = rasterize(Path(f"reference/svg/{sid}.svg").read_text(encoding="utf-8"), 850, None)
    ph = Image.open(f"reference/photos/{sid}.png").convert("RGB")
    ph.thumbnail((425, 850))
    img = Image.new("RGB", (425 + 600 + 600 + 20, 850), "white")
    img.paste(ph, (0, 0))
    img.paste(Image.fromarray(a), (435, 0))
    img.paste(Image.fromarray(d), (1045, 0))
    img.save(out / "cur.png")
    Image.fromarray(rasterize(svg, 1700, None)).crop((300, 0, 900, 560)).save(out / "z.png")
    print(f"wrote {out / 'cur.png'} and {out / 'z.png'}")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output/review"))
