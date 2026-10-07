from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def list_rgb_frames(episode_dir: str | Path) -> list[Path]:
    rgb_dir = Path(episode_dir) / "rgb"
    frames = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    frames = sorted(set(frames))
    if not frames:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
    return frames


def read_depth_m(episode_dir: str | Path, stem: str, depth_unit_m: float = 0.001) -> np.ndarray | None:
    depth_dir = Path(episode_dir) / "depth"
    npy_path = depth_dir / f"{stem}.npy"
    png_path = depth_dir / f"{stem}.png"
    if png_path.exists():
        depth_raw = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            return None
        return depth_raw.astype(np.float32) * float(depth_unit_m)
    if npy_path.exists():
        depth = np.load(npy_path)
        return depth.astype(np.float32)
    return None


def read_episode_metadata(episode_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(episode_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}
    return read_json(metadata_path)
