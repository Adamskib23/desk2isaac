# Project Summary

## What I Built

I recorded pick-and-place demonstrations on my desk and used them to drive a simulated Panda arm. An overhead Orbbec Gemini 2 records colour and aligned depth. Four measured ArUco markers calibrate a rectangular workspace. Prompted SAM 2 masks isolate the selected object, then geometric checks produce its approximate centre and upright cuboid yaw.

The robot receives an observed start, observed goal and carry guidance. A scripted differential-IK controller performs approach, grasp, lift, carry, place and release. The moving object is a free rigid body with gravity and finger contact, not a visual attachment. Simulated markers and stationary objects help compare the video and robot views, but do not act as obstacles.

I also trained an RGB pose student using accepted depth-tracker observations as pseudo-labels. A frozen ResNet-18 feeds a small trained head alongside mask geometry and known object information. Prediction does not read depth or teacher labels, although it relies on prepared masks and saved calibration.

The main system is demonstration retargeting; the student is supervised perception. There is no learned robot policy, RL, VLA or world model in this submission.

## Where It Stands

The 30-clip physical review passed 29 tasks. All 20 cuboid tasks finished within 5 mm and passed upright yaw checks. The sphere was harder to release accurately: none met the tighter 5 mm diagnostic and one failed the 25 mm pass threshold.

The two-hour RGB run reached 5.24 mm median positional disagreement with approximate depth teachers on eight validation clips. Carry-phase agreement improved markedly over a rest-height baseline, but the baseline was better on many stationary frames and learned box yaw remained poor. The validation batch selected checkpoints, so these are not independent test results.

Only three prepared robot tasks are included publicly, alongside all 3,007 sampled RGB crops, labels, both trained heads, both training histories and the full 30-task review results. The raw full-resolution RGB-D captures, intermediate masks and development artifacts stay local. [Data](data.md) explains what can and cannot be reproduced from the public package.

## Main Decisions

Colour masking was quick to prototype but unreliable around the dark box's shadow. SAM masks improved object isolation. Explicit geometry and quality checks remained useful because a clean mask alone does not guarantee a correct metric pose.

I chose Isaac Lab because I already knew it and wanted contact-based grasp evidence. I stopped short of residual RL once cuboid execution was reliable and used the remaining experiment for depth-supervised perception. I have not claimed the small RGB dataset supports robust deployment or unseen-object generalisation.

The camera calibration has an unresolved scale discrepancy against a rough ruler measurement. The scene assumes mass/friction and upright cuboid geometry. Orientation is yaw only, and the tin is approximated as a cuboid. These limits matter more than adding another model label to the project.

## Reading Order

Start with the [README](../README.md), then the [pipeline sheet](pipelines.md). [Setup](setup.md) contains runnable commands. [Results](results.md) connects each headline number to reports, videos and figures. [RGB experiment](rgb_perception.md) explains the learning recipe in detail.

The earlier development write-ups and pre-training baseline were preserved locally under `development/archive/`; they are not part of the public submission.

## Before Submission

- [x] Keep the working RGB-D retargeting and physical controller.
- [x] Keep the supervised experiment, both trained heads and its limitations.
- [x] Write compact pipeline, setup, data and evidence documents.
- [x] Include the existing screenshot and recorded paired examples.
- [x] Exclude raw captures, downloaded models, environments, intermediate output and local history.
- [x] Add MIT code licence and document separate data/third-party terms.
- [x] Verify the clean export and inspect its README/media locally.
- [ ] Create the public GitHub repository and push the clean package.
- [ ] Check the repository is accessible while signed out, then submit its URL by 9 October 2026, 23:59 BST.

No further model training or data collection is required for the current submission scope. GitHub publishing and application submission are still separate actions.
