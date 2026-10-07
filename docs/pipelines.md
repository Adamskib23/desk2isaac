# Pipeline Sheet

## 1. Demonstration Retargeting

**In plain English:** I move an object on my desk. The camera records colour and depth. The software measures where the object started, where it ended and how it travelled. A simulated robot then performs its own grasp and follows that guidance.

```text
Personal RGB-D recording + camera metadata
                 |
Four flat ArUco markers -> saved desk calibration
                 |
Prompted SAM 2 masks + aligned depth + object dimensions
                 |
Quality-gated centre positions and upright cuboid yaw
                 |
Observed start/goal + smoothed carry guidance
                 |
Desk-to-simulation mapping -> approach/grasp/carry/place plan
                 |
Panda differential IK + finger actuation + free-object physics
                 |
Paired review videos and contact/placement/yaw metrics
```

| Stage | Technical detail | Plain-English meaning |
| --- | --- | --- |
| Capture | Approximately 30 Hz aligned RGB-D, saved intrinsics, distortion, depth scale and timestamps | Each colour frame has matching distance measurements |
| Calibration | Markers 0-3, `DICT_4X4_1000`, 106.4 mm outer black squares; camera-to-desk transform and an inset rectangular workspace | Express positions in metres on the desk rather than image pixels |
| Segmentation | An operator selects the object; SAM 2.1 Tiny propagates its mask through the clip | Tell the tracker which object to follow, including when a hand partly covers it |
| Position | Eroded mask, valid foreground above the desk, connected-component and extent checks; cuboid height/ray estimate or known-radius sphere fit | Remove table pixels and reject implausible measurements before estimating the centre |
| Yaw | Fit the known rectangular upper face in metric XY; gate poor fits, tilted tops and ambiguous axes | Estimate how an upright box or tin is turned; do not assign an orientation to a featureless sphere |
| Trajectory | Require observed boundary windows; smooth guidance and record excluded frames | The robot gets a stable start, destination and useful path, without treating interpolated positions as measurements |
| Retargeting | Desk XY rotation/translation into the Panda workspace; measured object size; adjusted clearance and grasp offset | Adapt the human motion to a top-down robot grasp |
| Execution | Scripted phases, differential IK and finger commands; manipulated rigid body remains free | The fingers must actually hold the object against gravity |
| Review | Contact, lift, slip, release, settle and final XY checks; cuboid yaw and tilt checks | A visually plausible replay is not enough to count as a successful grasp |

Yaw represents an undirected shape axis, modulo 180 degrees. It is not complete six-degree-of-freedom pose. Missing yaw observations may be filled for control, not used as observed training labels.

The controller receives start, goal and carry guidance. It adds approach, closing, lifting, lowering, opening and settling phases. A minimum carry clearance changes the executed path, so tracking metrics are relative to that controller reference, not an exact replay of the human motion.

Four simulated marker patterns and the stationary objects provide visual reference. They have collisions disabled. The moving object has collisions enabled; its pose is not commanded during the run.

**Prerequisites:** existing clips can be replayed with Isaac Lab, its Panda assets and an NVIDIA GPU. Making new trajectories additionally needs an RGB-D camera, unchanged/calibrated marker layout, measured object dimensions, OpenCV and SAM 2. See [Setup](setup.md).

**Included entry point:** `python scripts/replay_demo.py`. It loads three prepared real-data tasks and frozen visual scene snapshots, rather than rebuilding the omitted raw captures.

## 2. Depth-supervised RGB Perception

**In plain English:** the depth tracker supplies approximate answers during training. A small learned model tries to predict those answers from colour images and object-mask geometry. Depth is not opened when making predictions. This is a separate experiment, not the robot's controller.

```text
Same personally recorded clips
      |                          |
RGB + saved mask geometry        Aligned depth + geometric tracker
      |                          |
224 x 224 crops + 15 features    Observed pose candidates only
      |                          |
      +-------- supervised fitting --------+
                       |
Frozen ImageNet ResNet-18 -> trained 128/64-unit MLP head
                       |
Height + image-centre offsets + doubled-angle yaw
                       |
Saved camera calibration -> desk XYZ and yaw candidates
                       |
RGB-only prediction -> separate teacher-agreement evaluation
```

| Stage | Technical detail | Plain-English meaning |
| --- | --- | --- |
| Split | Whole recordings and recording batches grouped; 22 train clips, 8 validation clips | Adjacent frames from one clip cannot leak into both sides |
| Supervision | Approximate observed depth centres and gated yaw; separate label file | Train against actual tracker observations, not its filled-in robot path |
| Inputs | RGB crop, mask shape, calibrated ray coordinates, known size and object identity | Depth, target pose, frame time and clip ID are not model features |
| Augmentation | Training-only brightness, contrast, colour and occasional blur | Vary appearance without changing metric labels |
| Model | Frozen 512-feature ResNet-18 encoder; normalised features plus geometry; five-output MLP | Reuse image features and learn a small task-specific prediction head |
| Loss | Huber position-target loss plus masked doubled-yaw MSE | Learn height and offsets; only train yaw where an orientation was observed |
| Prediction | Saved calibration maps the predicted height and centre ray into desk XYZ | Recover metric pose candidates using colour inputs and the known camera setup |
| Evaluation | Compare predictions to separate approximate teachers on the validation batch | Measure teacher agreement, not independently surveyed physical accuracy |

The public RGB bundle contains 3,007 sampled frames: 2,202 training and 805 validation. Both trained heads and their reports are included. Reversing a per-frame video adds no new supervised examples, so it was not used as augmentation.

**Prerequisites:** PyTorch, Torchvision and the included crops/geometry/calibration. The compact checkpoints download the frozen ImageNet encoder on first use. No camera, depth file, SAM installation or Isaac Lab is needed for this bundled inference experiment. Rebuilding its labels needs the original RGB-D recordings, which are not included.

This is not yet an end-to-end raw-RGB pipeline. Predictions consume cached RGB crops and mask-derived features; initial mask prompting in dataset preparation could use depth proposals. Neither live RGB-only mask generation nor learned-pose robot execution was evaluated.

**Included entry points:** `scripts/predict_rgb_student.py`, `scripts/evaluate_rgb_student.py` and `scripts/train_rgb_student.py`. The [RGB experiment](rgb_perception.md) documents the recipes and limitations.
