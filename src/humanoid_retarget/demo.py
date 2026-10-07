from __future__ import annotations

from pathlib import Path

from humanoid_retarget.io import read_json
from humanoid_retarget.mask_tracking import file_sha256
from humanoid_retarget.scene_context import load_scene_context


def bundled_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Demo paths must remain inside the bundle")
    return path


def load_demo(path: Path) -> dict:
    manifest = read_json(path)
    if manifest.get("schema_version") != "hmnd_portable_demo_v1":
        raise ValueError("Unsupported demo manifest")
    indices = [row["index"] for row in manifest["clips"]]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Demo clip indices must be nonempty and unique")
    for row in manifest["clips"]:
        task = bundled_path(path.parent, row["trajectory"])
        scene = bundled_path(path.parent, row["scene_context"])
        for source, expected in ((task, row["trajectory_sha256"]), (scene, row["scene_context_sha256"])):
            if file_sha256(source) != expected:
                raise ValueError(f"Packaged demo file changed: {source.name}")
        if read_json(task)["object_name"] != row["object_name"]:
            raise ValueError("Demo object does not match trajectory")
        load_scene_context(scene, task)
    return manifest
