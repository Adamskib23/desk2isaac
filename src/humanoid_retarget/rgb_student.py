from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.rgb_data import GEOMETRY_NAMES, decode_pose
from humanoid_retarget.mask_tracking import file_sha256


class RGBPoseStudent(nn.Module):
    """Frozen RGB encoder plus a trained metric-height/centre/yaw head."""

    def __init__(self, *, pretrained: bool = False):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.encoder.requires_grad_(False)
        self.register_buffer("feature_mean", torch.zeros(512))
        self.register_buffer("feature_std", torch.ones(512))
        self.register_buffer("geometry_mean", torch.zeros(len(GEOMETRY_NAMES)))
        self.register_buffer("geometry_std", torch.ones(len(GEOMETRY_NAMES)))
        self.head = nn.Sequential(nn.Linear(512 + len(GEOMETRY_NAMES), 128), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 5))

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def from_features(self, features, geometry):
        values = torch.cat(((features - self.feature_mean) / self.feature_std,
                            (geometry - self.geometry_mean) / self.geometry_std), dim=1)
        return self.head(values)

    def forward(self, images, geometry):
        return self.from_features(self.encoder(images), geometry)


def image_tensor(path: Path, rng: np.random.Generator | None = None) -> torch.Tensor:
    image = cv2.imread(str(path))
    if image is None or image.shape != (224, 224, 3):
        raise ValueError("Student expects a 224x224 RGB crop")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    if rng is not None:
        # Photometric changes preserve pixel geometry and metric labels.
        rgb = (rgb - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.12, 0.12)
        rgb *= rng.uniform(0.85, 1.15, size=(1, 1, 3))
        if rng.random() < 0.3:
            rgb = cv2.GaussianBlur(rgb, (3, 3), 0.7)
        rgb = np.clip(rgb, 0, 1)
    rgb = (rgb - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


def pose_loss(prediction, targets, yaw_observed):
    position = nn.functional.smooth_l1_loss(prediction[:, :3], targets[:, :3], beta=0.1)
    yaw = ((prediction[:, 3:] - targets[:, 3:]) ** 2).mean(dim=1)
    yaw = (yaw * yaw_observed).sum() / yaw_observed.sum().clamp_min(1)
    return position + 0.05 * yaw


def export_student_head(source: Path, encoder_weights: Path, destination: Path) -> None:
    """Package the trained head only after verifying the frozen upstream encoder."""
    if destination.exists():
        raise ValueError("Choose a fresh checkpoint destination")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "hmnd_rgb_pose_student_v1":
        raise ValueError("Expected a full RGB student checkpoint")
    if checkpoint["geometry_feature_names"] != list(GEOMETRY_NAMES):
        raise ValueError("Checkpoint geometry feature order changed")
    model = RGBPoseStudent()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    upstream = torch.load(encoder_weights, map_location="cpu", weights_only=True)
    upstream_encoder = resnet18(weights=None)
    upstream_encoder.fc = nn.Identity()
    # Torchvision supplies zero batch counters absent from older ImageNet weight files.
    upstream_encoder.load_state_dict({k: v for k, v in upstream.items() if not k.startswith("fc.")}, strict=True)
    for name, value in model.encoder.state_dict().items():
        if not torch.equal(value, upstream_encoder.state_dict()[name]):
            raise ValueError(f"Encoder differs from upstream weights: {name}")
    compact = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    compact.update(
        schema_version="hmnd_rgb_pose_student_head_v1",
        encoder="torchvision_resnet18_IMAGENET1K_V1_frozen",
        encoder_weights_sha256=file_sha256(encoder_weights),
        source_checkpoint_sha256=file_sha256(source),
        state_dict={k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("encoder.")},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, destination)


def load_student(path: Path, device: str, *, encoder_cache: Path | None = None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    schema = checkpoint.get("schema_version")
    if schema not in {"hmnd_rgb_pose_student_v1", "hmnd_rgb_pose_student_head_v1"}:
        raise ValueError("Unsupported RGB student checkpoint")
    if checkpoint["geometry_feature_names"] != list(GEOMETRY_NAMES):
        raise ValueError("Checkpoint geometry feature order changed")
    if schema == "hmnd_rgb_pose_student_v1":
        model = RGBPoseStudent()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        if checkpoint.get("encoder") != "torchvision_resnet18_IMAGENET1K_V1_frozen":
            raise ValueError("Unsupported frozen RGB encoder")
        if encoder_cache is not None:
            torch.hub.set_dir(str(encoder_cache))
        model = RGBPoseStudent(pretrained=True)
        weights_name = Path(urlparse(ResNet18_Weights.IMAGENET1K_V1.url).path).name
        weights_path = Path(torch.hub.get_dir()) / "checkpoints" / weights_name
        if file_sha256(weights_path) != checkpoint["encoder_weights_sha256"]:
            raise ValueError("Frozen encoder weights do not match the exported checkpoint")
        expected = {k for k in model.state_dict() if not k.startswith("encoder.")}
        if set(checkpoint["state_dict"]) != expected:
            raise ValueError("Compact checkpoint head/normalisation keys do not match")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model.to(device).eval(), checkpoint


def predict_inputs(model, inputs: dict, root: Path, *, split: str, device: str, batch_size: int = 64):
    samples = [r for r in inputs["samples"] if split == "all" or r["split"] == split]
    if not samples or batch_size < 1:
        raise ValueError("Choose a nonempty split and a positive batch size")
    calibrations = {k: WorkspaceCalibration(**v) for k, v in inputs["calibrations"].items()}
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            rows = samples[start:start + batch_size]
            images = torch.stack([image_tensor(root / r["rgb_crop"]) for r in rows]).to(device)
            geometry = torch.tensor([r["geometry"] for r in rows], dtype=torch.float32, device=device)
            outputs = model(images, geometry).cpu().numpy()
            for row, output in zip(rows, outputs):
                predictions.append({"sample_id": row["sample_id"], "clip_index": row["clip_index"],
                                    "frame_idx": row["frame_idx"], "object_name": row["object_name"],
                                    **decode_pose(output, row, calibrations[row["calibration_key"]])})
    return predictions
