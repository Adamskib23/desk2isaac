from pathlib import Path
import runpy
import shutil

import pytest

from humanoid_retarget.demo import bundled_path, load_demo
from humanoid_retarget.io import write_json
from humanoid_retarget.mask_tracking import file_sha256


def make_bundle(root):
    task = root / "box/trajectory.json"
    scene = root / "box/scene_context.json"
    write_json(task, {"object_name": "box", "source_episode": "data/real/original_box"})
    write_json(scene, {"schema_version": "hmnd_scene_context_v1", "selected_object": "box",
                      "source_episode": "data/real/original_box", "source_task_sha256": file_sha256(task),
                      "source_files_relative_to": "context",
                      "source_file_sha256": {"trajectory.json": file_sha256(task)},
                      "collision_enabled": False, "background_objects": []})
    return {"schema_version": "hmnd_portable_demo_v1", "clips": [
        {"index": 23, "object_name": "box", "trajectory": "box/trajectory.json",
         "scene_context": "box/scene_context.json", "trajectory_sha256": file_sha256(task),
         "scene_context_sha256": file_sha256(scene)}]}


def test_demo_survives_relocation_without_original_recordings(tmp_path, monkeypatch):
    root = tmp_path / "original"
    write_json(root / "manifest.json", make_bundle(root))
    destination = tmp_path / "clone"
    shutil.copytree(root, destination)
    shutil.rmtree(root)
    monkeypatch.chdir(tmp_path)
    assert load_demo(destination / "manifest.json")["clips"][0]["index"] == 23


@pytest.mark.parametrize("problem", ["schema", "empty", "duplicate", "hash", "name", "escape"])
def test_demo_rejects_wrong_or_changed_files(tmp_path, problem):
    manifest = make_bundle(tmp_path)
    if problem == "schema":
        manifest["schema_version"] = "unknown"
    elif problem == "empty":
        manifest["clips"] = []
    elif problem == "duplicate":
        manifest["clips"] *= 2
    elif problem == "hash":
        manifest["clips"][0]["trajectory_sha256"] = "changed"
    elif problem == "name":
        manifest["clips"][0]["object_name"] = "sphere"
    else:
        manifest["clips"][0]["trajectory"] = "../escape.json"
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError):
        load_demo(tmp_path / "manifest.json")


def test_bundled_path_rejects_absolute_and_symlink_escape(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "outside").symlink_to(tmp_path / "external")
    for value in (str(tmp_path / "file.json"), "outside/file.json", "../file.json"):
        with pytest.raises(ValueError):
            bundled_path(root, value)


@pytest.fixture
def checker():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_submission.py"))


def test_document_links_are_relative_to_the_document(tmp_path, checker):
    write_json(tmp_path / "present.json", {})
    doc = tmp_path / "README.md"
    doc.write_text("[present](present.json) [external](https://example.org) [heading](#result)")
    checker["check_links"](doc)
    doc.write_text("[missing](missing.json)")
    with pytest.raises(ValueError, match="Broken documentation link"):
        checker["check_links"](doc)


def test_inventory_detects_changed_and_extra_files(tmp_path, checker):
    file = tmp_path / "README.md"
    file.write_text("Submission")
    original = {"schema_version": "hmnd_submission_files_v1", "files": {
        "README.md": {"sha256": file_sha256(file), "bytes": file.stat().st_size}}}
    write_json(tmp_path / "SUBMISSION_MANIFEST.json", original)
    checker["check_inventory"](tmp_path, strict=True)
    (tmp_path / "unlisted.txt").write_text("unlisted")
    with pytest.raises(ValueError, match="inventory differs"):
        checker["check_inventory"](tmp_path, strict=True)
    (tmp_path / "unlisted.txt").unlink()
    file.write_text("Changed")
    with pytest.raises(ValueError, match="Exported file changed"):
        checker["check_inventory"](tmp_path, strict=False)


def test_example_model_heads_are_not_treated_as_runtime_artifacts(tmp_path, checker):
    write_json(tmp_path / "examples/models/head.json", {})
    write_json(tmp_path / "models/cache.json", {})
    assert set(checker["included_files"](tmp_path)) == {"examples/models/head.json"}
