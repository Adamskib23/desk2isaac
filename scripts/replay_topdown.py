#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from humanoid_retarget.io import read_json
from humanoid_retarget.visualization import render_topdown_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a quick top-down replay of a retargeted trajectory.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out", default=None, help="Output MP4 path.")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    trajectory_path = Path(args.trajectory)
    trajectory = read_json(trajectory_path)
    out = Path(args.out) if args.out else trajectory_path.with_name("topdown_replay.mp4")
    image_path = out.with_suffix(".png")
    render_topdown_replay(trajectory, out, image_path=image_path, fps=args.fps)
    print(f"Wrote {out}")
    print(f"Wrote {image_path}")


if __name__ == "__main__":
    main()

