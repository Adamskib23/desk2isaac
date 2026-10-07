from __future__ import annotations

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from humanoid_retarget.calibration import WorkspaceCalibration
from humanoid_retarget.rgb_data import GEOMETRY_NAMES
from humanoid_retarget.rgb_student import RGBPoseStudent, export_student_head, image_tensor, load_student, pose_loss, predict_inputs


def test_frozen_encoder_batch_norm_stays_frozen():
    model = RGBPoseStudent().train()
    assert not model.encoder.training
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    prediction = model.from_features(torch.randn(2, 512), torch.randn(2, len(GEOMETRY_NAMES)))
    prediction.sum().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.head.parameters())


def test_position_only_examples_do_not_train_yaw():
    prediction = torch.zeros(2, 5, requires_grad=True)
    target = torch.ones(2, 5)
    loss = pose_loss(prediction, target, torch.zeros(2))
    loss.backward()
    assert torch.equal(prediction.grad[:, 3:], torch.zeros(2, 2))
    assert torch.isfinite(loss)


def test_checkpoint_load_needs_no_pretrained_download(tmp_path):
    model = RGBPoseStudent()
    path = tmp_path / "best.pt"
    torch.save({"schema_version": "hmnd_rgb_pose_student_v1", "state_dict": model.state_dict(),
                "geometry_feature_names": list(GEOMETRY_NAMES)}, path)
    loaded, _ = load_student(path, "cpu")
    assert not loaded.training
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_compact_checkpoint_matches_full_model_without_packaging_encoder(tmp_path, monkeypatch):
    import humanoid_retarget.rgb_student as module
    from torchvision.models import resnet18
    torch.manual_seed(7)
    encoder = resnet18(weights=None)
    upstream = tmp_path / "checkpoints/resnet18-f37072fd.pth"
    upstream.parent.mkdir()
    torch.save(encoder.state_dict(), upstream)
    model = RGBPoseStudent()
    model.encoder.load_state_dict({k: v for k, v in encoder.state_dict().items() if not k.startswith("fc.")})
    source, compact = tmp_path / "full.pt", tmp_path / "head.pt"
    torch.save({"schema_version": "hmnd_rgb_pose_student_v1", "state_dict": model.state_dict(),
                "geometry_feature_names": list(GEOMETRY_NAMES)}, source)
    export_student_head(source, upstream, compact)
    assert compact.stat().st_size < source.stat().st_size / 50

    def cached_resnet(*, weights):
        assert weights is not None
        restored = resnet18(weights=None)
        restored.load_state_dict(encoder.state_dict())
        return restored

    monkeypatch.setattr(module, "resnet18", cached_resnet)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    loaded, checkpoint = load_student(compact, "cpu")
    assert not any(k.startswith("encoder.") for k in checkpoint["state_dict"])
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    checkpoint["state_dict"]["encoder.fake"] = torch.zeros(1)
    torch.save(checkpoint, tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="keys do not match"):
        load_student(tmp_path / "invalid.pt", "cpu")
    upstream.write_bytes(b"changed")
    with pytest.raises(ValueError, match="weights do not match"):
        load_student(compact, "cpu")


def test_head_export_rejects_a_changed_encoder(tmp_path):
    model = RGBPoseStudent()
    source, weights = tmp_path / "full.pt", tmp_path / "encoder.pt"
    torch.save({"schema_version": "hmnd_rgb_pose_student_v1", "state_dict": model.state_dict(),
                "geometry_feature_names": list(GEOMETRY_NAMES)}, source)
    changed = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    changed["conv1.weight"].add_(1)
    torch.save(changed, weights)
    with pytest.raises(ValueError, match="Encoder differs"):
        export_student_head(source, weights, tmp_path / "head.pt")


def test_rgb_prediction_does_not_open_teacher_files(tmp_path, monkeypatch):
    crop_path = tmp_path / "crop.jpg"
    assert cv2.imwrite(str(crop_path), np.full((224, 224, 3), 128, np.uint8))
    calibration = WorkspaceCalibration((200, 160), ((0, 0), (199, 0), (199, 159), (0, 159)),
                                        ((0, 0), (0.4, 0), (0.4, -0.32), (0, -0.32)))
    from dataclasses import asdict
    row = {"sample_id": "a", "clip_index": 1, "frame_idx": 0, "split": "val", "rgb_crop": "crop.jpg",
           "geometry": [0.] * len(GEOMETRY_NAMES), "calibration_key": "1", "object_name": "blue_sphere",
           "size_m": [0.0365] * 3, "pixel_xy": [100, 80], "bbox_wh_px": [20, 20]}
    inputs = {"samples": [row], "calibrations": {"1": asdict(calibration)}}

    class ConstantModel(torch.nn.Module):
        def forward(self, image, geometry):
            return torch.tensor([[1., 0., 0., 1., 0.]]).repeat(len(image), 1)

    import humanoid_retarget.rgb_student as module
    original = module.image_tensor

    def guard(path, rng=None):
        assert path == crop_path
        return original(path, rng)

    monkeypatch.setattr(module, "image_tensor", guard)
    predicted = predict_inputs(ConstantModel(), inputs, tmp_path, split="val", device="cpu")
    assert predicted[0]["desk_xyz_m"][2] == pytest.approx(0.1)
    assert predicted[0]["desk_yaw_rad"] is None


def test_photometric_augmentation_seeded_and_validation_unmodified(tmp_path):
    path = tmp_path / "crop.jpg"
    assert cv2.imwrite(str(path), np.full((224, 224, 3), 128, np.uint8))
    base = image_tensor(path)
    augmented = image_tensor(path, np.random.default_rng(42))
    assert torch.equal(augmented, image_tensor(path, np.random.default_rng(42)))
    assert torch.equal(base, image_tensor(path))
    assert not torch.equal(base, augmented)


@pytest.fixture
def training_functions():
    import runpy
    from pathlib import Path
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/train_rgb_student.py"))


def test_periodic_checkpoint_survives_failed_replacement(tmp_path, monkeypatch, training_functions):
    save = training_functions["save_progress"]
    checkpoint = {"best_epoch": 1, "state_dict": {"test": torch.ones(1)}}
    save(tmp_path, checkpoint, [{"epoch": 1}], {"running": True})
    assert torch.load(tmp_path / "best.pt", weights_only=True)["best_epoch"] == 1

    def fail_save(payload, path):
        path.write_bytes(b"incomplete")
        raise OSError("disk write interrupted")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="interrupted"):
        save(tmp_path, {"best_epoch": 2}, [], {})
    assert torch.load(tmp_path / "best.pt", weights_only=True)["best_epoch"] == 1


def test_augmentation_refresh_encodes_only_requested_training_rows(tmp_path, monkeypatch, training_functions):
    from types import SimpleNamespace
    function = training_functions["encode_features"]
    opened = []

    def image(path, rng):
        opened.append(path.name)
        assert rng is not None
        return torch.zeros(3, 4, 4)

    monkeypatch.setitem(function.__globals__, "image_tensor", image)
    model = SimpleNamespace(encoder=lambda images: torch.ones(len(images), 512))
    features = function(model, [0], [{"rgb_crop": "train.jpg"}, {"rgb_crop": "val.jpg"}],
                        tmp_path, 1, "cpu", np.random.default_rng(42))
    assert features.shape == (1, 512)
    assert opened == ["train.jpg"]


def test_augmentation_refresh_stops_before_loading_when_budget_expired(tmp_path, training_functions):
    from types import SimpleNamespace
    result = training_functions["encode_features"](None, [0], [], tmp_path, 1, "cpu",
                                                    budget=SimpleNamespace(time_expired=True))
    assert result is None
