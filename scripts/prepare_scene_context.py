#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from humanoid_retarget.dataset_review import load_review_manifest
from humanoid_retarget.io import write_json
from humanoid_retarget.scene_context import prepare_scene_context


def main():
    parser = argparse.ArgumentParser(description="Prepare marker/background visual references without launching Isaac.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/datasets/pilot_030/manifest.json")
    parser.add_argument("--objects-config", type=Path, default=ROOT / "configs/objects.rl_pilot.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+")
    args = parser.parse_args()
    directory = args.out.resolve()
    if (not directory.is_relative_to(ROOT) or directory.is_relative_to(ROOT / "data")
            or directory.is_relative_to(args.manifest.parent.resolve())
            or (directory.exists() and any(directory.iterdir()))):
        parser.error("Choose a fresh repository output directory outside captures/the frozen dataset")
    try:
        manifest = load_review_manifest(args.manifest, ROOT)
        chosen = set(args.indices or range(1, manifest["clip_count"] + 1))
        if not chosen.issubset(range(1, manifest["clip_count"] + 1)):
            parser.error("Indices must be within the dataset")
        rows = []
        for row in manifest["clips"]:
            if row["index"] not in chosen:
                continue
            path = directory / f"{row['index']:03d}_{row['object_name']}" / "scene_context.json"
            context = prepare_scene_context(ROOT / row["trajectory"], ROOT / row["calibration"]["config"],
                                            args.objects_config, path)
            rows.append({"index": row["index"], "episode": row["episode"], "context": str(path.relative_to(ROOT)),
                         "markers": len(context["markers"]), "background_objects": [o["name"] for o in context["background_objects"]],
                         "diagnostics": context["background_diagnostics"]})
            print(f"SCENE {row['index']:02d}: {len(context['markers'])} markers, backgrounds={rows[-1]['background_objects']}", flush=True)
        write_json(directory / "contexts.json", {"schema_version": "hmnd_scene_context_inventory_v1", "clips": rows,
                                                  "notes": "Preview-only references. Source recordings and frozen tasks unchanged."})
    except (ValueError, OSError) as error:
        parser.exit(1, f"Cannot prepare scene references: {error}\n")


if __name__ == "__main__":
    main()
