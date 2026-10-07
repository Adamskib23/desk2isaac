#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.demo import bundled_path, load_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three packaged real-data tasks using physical grasps")
    parser.add_argument("--manifest", type=Path, default=ROOT / "examples/retargeting/manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/demo")
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Check files and print commands without Isaac Lab")
    args = parser.parse_args()
    manifest = load_demo(args.manifest)
    selected = args.indices or [row["index"] for row in manifest["clips"]]
    available = {row["index"]: row for row in manifest["clips"]}
    if len(selected) != len(set(selected)) or any(i not in available for i in selected):
        parser.error("Select unique indices from the demo manifest")
    output = args.out.resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        parser.error("Choose a fresh directory under outputs")
    if not args.dry_run:
        output.mkdir(parents=True)
    for index in selected:
        row = available[index]
        directory = output / f"{index:03d}_{row['object_name']}"
        command = ["bash", str(ROOT / "scripts/run_physical_grasp.sh"),
                   "--trajectory", str(bundled_path(args.manifest.parent, row["trajectory"])),
                   "--scene-context", str(bundled_path(args.manifest.parent, row["scene_context"])),
                   "--report", str(directory / "metrics.json")]
        if args.headless:
            command.append("--headless")
        if args.video:
            command.extend(["--video", str(directory / "simulation.mp4")])
        print(f"Clip {index}: {row['object_name']}", flush=True)
        if args.dry_run:
            print(" ".join(command))
        else:
            directory.mkdir()
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
