"""One-time download of SAM 2.1 weights into models/. The only file in the repo that uses the network."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rosette.perception.sam2 import REPOS, local_dir  # noqa: E402


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=[*REPOS, "all"], required=True)
    ap.add_argument("--dir", type=Path, default=ROOT / "models")
    args = ap.parse_args(argv)
    from huggingface_hub import snapshot_download

    for size in (REPOS if args.model == "all" else [args.model]):
        target = local_dir(args.dir, size)
        snapshot_download(
            REPOS[size],
            local_dir=target,
            allow_patterns=["*.json", "*.safetensors", "README.md"],
        )
        print(f"{target}  {folder_size(target) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
