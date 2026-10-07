# Setup and Running

Run commands from the repository root. Keep capture, ML and Isaac environments separate: the simulator brings its own dependency stack. None of the included demos needs a connected camera.

## CPU Checks

Python 3.11 was used for the ML and simulator runs; the base package requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/check_submission.py
python scripts/replay_demo.py --dry-run
```

Torch-specific tests skip if Torch is not installed. The submission checker verifies bundled hashes, whole-recording splits, evidence counts, local documentation links and media files. It does not launch Isaac Lab.

A quick 2D view of an included real-data trajectory:

```bash
mkdir -p outputs
python scripts/replay_topdown.py \
  --trajectory examples/retargeting/023_black_box/trajectory.json \
  --out outputs/box_topdown.mp4
```

This plot is a trajectory visualisation, not a physics test.

## Isaac Lab Replay

The review ran on Ubuntu with an RTX 5060 Laptop GPU (8 GB), Python 3.11.14, Isaac Sim package `5.1.0.0`, Isaac Lab package `0.53.1`, and PyTorch `2.7.0+cu128`. These are the installed package versions, not a claim that arbitrary versions are interchangeable.

Install Isaac Lab and a compatible Isaac Sim using the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Check NVIDIA's driver/GPU requirements. The first scene launch also needs access to Isaac Lab's configured asset server, or locally cached Panda and ground-plane assets. This repository does not bundle them. A fresh simulator installation has not been tested end to end.

```bash
export ISAAC_PYTHON=/path/to/isaaclab/environment/bin/python
python scripts/replay_demo.py --out outputs/demo
```

The script runs box, sphere and tin tasks sequentially with physical contact grasps. The GUI shows the simulation; the paired source/simulation videos are already saved in [the README](../README.md#watch). Since full-resolution source recordings are omitted, the compact demo does not generate a new paired source view.

To save headless simulation videos and reports:

```bash
python scripts/replay_demo.py --headless --video --out outputs/demo_headless
```

To run only the box, add `--indices 23`. Each task writes a `metrics.json` under the output directory. Do not use `--mode kinematic` as evidence of a grasp: that older diagnostic explicitly repositions the object. The demo wrapper selects dynamic mode.

With your own full recordings, `scripts/replay_dataset.py` runs a prepared playlist and uses `--reference-episode` internally to display the source alongside the simulation. Its full-review manifests require the original RGB-D files and masks; the compact demo manifest is a different format.

## RGB Prediction

Create a separate ML environment. For CPU-only evaluation:

```bash
python3.11 -m venv .venv-ml
source .venv-ml/bin/activate
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[ml,dev]'
python -m pytest -q tests/test_rgb_student.py tests/test_training_control.py
python scripts/predict_rgb_student.py \
  --inputs examples/rgb_supervision/inputs.json \
  --checkpoint examples/models/rgb_long_head.pt \
  --device cpu --out outputs/rgb_predictions.json
python scripts/evaluate_rgb_student.py \
  --manifest examples/rgb_supervision/training_manifest.json \
  --predictions outputs/rgb_predictions.json \
  --out outputs/rgb_evaluation --plots
```

For the original GPU recipe, install the same Torch/Torchvision versions from `https://download.pytorch.org/whl/cu128` instead of the CPU index. This version pairing and the CPU/CUDA indexes are listed in [PyTorch's previous-version instructions](https://pytorch.org/get-started/previous-versions/). Verify CUDA is available before using `--device cuda`. Do not upgrade the shared Isaac environment to install these dependencies.

The compact heads include the learned layers and normalisation buffers. Torchvision downloads the official frozen ResNet-18 encoder on first use, and its complete file hash is checked. An existing cache can be supplied with `--encoder-cache /path/to/torch/hub`; it must contain `checkpoints/resnet18-f37072fd.pth`. Full training checkpoints remain supported too.

Prediction only opens `inputs.json`, RGB crops and model files. Evaluation separately opens `teacher_labels.json`. Small floating-point differences between CPU inference and cached GPU feature evaluation are expected. [Training recipes](rgb_perception.md#training-commands) reproduce the experiment settings; no further training is required to inspect the saved results.

## New RGB-D Data

This section is for extending the project. The original physical marker layout has moved; do not reuse its saved calibration for new recordings.

Use an overhead Gemini 2, supported Orbbec Python bindings, flat ArUco markers 0-3 from `DICT_4X4_1000`, and measured object dimensions. The black square side in this setup was 106.4 mm, excluding the white border. Keep all four markers visible and check scale with a second measured distance.

```bash
python -m pip install -e '.[orbbec]'
python scripts/record_orbbec_episode.py \
  --out data/real/new_calibration --seconds 3
python scripts/calibrate_workspace.py \
  --episode data/real/new_calibration --marker-size-mm 106.4 \
  --marker-ids 0 1 2 3 --dictionary DICT_4X4_1000 \
  --workspace-shape rectangle --out outputs/new_workspace.yaml
python scripts/record_orbbec_episode.py \
  --out data/real/new_box --seconds 10 --guided
```

Check the recorder reports depth aligned to colour, and that the metadata includes intrinsics, distortion and depth units. A generic OpenCV webcam recording lacks the calibrated RGB-D required by the SAM tracking path. Hold the object still at the beginning and end, lift it upright, then move and release it within the workspace.

### SAM 2

The recorded dataset used SAM 2.1 Tiny at source commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`. In a separate Torch-enabled environment, install the upstream source; the optional custom CUDA extension was disabled in this project:

```bash
mkdir -p third_party models/sam2
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -e third_party/sam2
curl -L --fail \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  -o models/sam2/sam2.1_hiera_tiny.pt
sha256sum models/sam2/sam2.1_hiera_tiny.pt
```

Expected checkpoint SHA-256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`. The [upstream README](https://github.com/facebookresearch/sam2) documents SAM's dependencies and model terms.

Select the box interactively in the first frame, inspect the propagated masks, then derive a trajectory:

```bash
python scripts/segment_sam2_episode.py \
  --episode data/real/new_box --object-name black_box \
  --out outputs/new_box_masks
python scripts/build_trajectory.py \
  --episode data/real/new_box --object-name black_box \
  --workspace-config outputs/new_workspace.yaml \
  --objects-config configs/objects.rl_pilot.yaml \
  --mask-dir outputs/new_box_masks --with-yaw \
  --out outputs/new_box_tracking/trajectory.json
bash scripts/run_physical_grasp.sh \
  --trajectory outputs/new_box_tracking/trajectory.json \
  --report outputs/new_box_metrics.json
```

For different objects, update the measured size and shape rather than only changing the label. The tin is approximated as a cuboid, not a detailed CAD mesh. Review tracking diagnostics and boundary observations before running the robot; generated pose candidates are not certified ground truth.
