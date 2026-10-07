# Depth-supervised RGB Experiment

The aim was modest: use my recorded depth data as approximate supervision, then predict object-centre pose from RGB inputs without opening depth at prediction time. I kept this separate from the robot controller so an uncertain learned estimate could not silently replace the evaluated tracker.

## Model

The image encoder is ImageNet ResNet-18, frozen in evaluation mode, including its batch-normalisation statistics. It produces 512 features from a 224 x 224 crop. Fifteen additional features describe the RGB mask, calibrated ray coordinates, known object dimensions and identity.

Training-set means and standard deviations normalise both inputs. A 128-unit layer, ReLU, 0.1 dropout, 64-unit layer, ReLU and five-output layer predict:

1. Object-centre height, scaled by 0.1 m.
2. Horizontal image-centre offset, scaled by bounding-box width.
3. Vertical image-centre offset, scaled by bounding-box height.
4. Cosine of twice the upright yaw.
5. Sine of twice the upright yaw.

Saved camera calibration maps the predicted centre ray and height into desk XYZ. Height is constrained not to put a known object below its nominal resting centre. Doubled-angle yaw represents a shape axis modulo 180 degrees; sphere yaw is undefined.

The encoder does not see depth, labels, motion phase, frame time or clip ID. Geometry features are calculated before fitting from RGB and the saved masks. The use of known size, identity and calibration makes this a constrained pose experiment, not general monocular scene understanding.

## Supervision and Fitting

The separate depth teacher file contains accepted observed position/yaw candidates. Interpolated robot guidance is excluded. Targets remain approximate pseudo-labels: this experiment measures agreement with the geometric tracker rather than independent physical accuracy.

The objective combines Huber loss on height/offset targets with 0.05 times masked doubled-yaw mean-squared error. Sphere and uncertain yaw observations do not contribute yaw loss. AdamW fits only the head, with weight decay 0.001 and batch size 64. Sampling balances object and initial/carry/final strata with replacement.

Whole-recording grouping gives 22 training clips and eight validation clips. Normalisation uses base training views only. Two additional photometric views change brightness, contrast, channel balance and occasional blur. There are no geometric flips or rotations requiring updated pose/calibration labels. Reversed videos would duplicate per-frame examples, so they were not used.

The best checkpoint minimises the combined validation loss. This repeatedly consulted validation set is not an independent test set.

## Training Commands

These commands document completed recipes, not a request to continue training. Set up a separate [ML environment](setup.md#rgb-prediction) first. Choose an unused output directory.

Short run:

```bash
python scripts/train_rgb_student.py \
  --manifest examples/rgb_supervision/training_manifest.json \
  --out outputs/ml/rgb_short_reproduction --allow-pseudo-labels \
  --epochs 80 --learning-rate 0.001 --augment-views 2 --seed 42
```

Two-hour run:

```bash
python scripts/train_rgb_student.py \
  --manifest examples/rgb_supervision/training_manifest.json \
  --out outputs/ml/rgb_long_reproduction --allow-pseudo-labels \
  --time-budget-minutes 120 --refresh-augment-every 20 \
  --learning-rate 0.0003 --augment-views 2 --seed 42
```

The time budget covers fitting, including augmentation refresh; initial feature extraction and final reporting are additional. The best full checkpoint and progress are saved periodically and at exit. CPU fitting is possible with `--device cpu`, but runtime/epoch count will differ. Exact wall-clock-limited reproduction is not guaranteed across hardware.

The 80-epoch run selected epoch 78. The two-hour run completed 12,880 epochs, refreshed augmentation 643 times and selected epoch 2,322. Its best checkpoint occurred about 21.9 minutes into fitting. Learning rate and augmentation refresh changed too, so the comparison does not isolate training duration.

## Saved Models and Results

Both compact heads in `examples/models/` contain the fitted layers and all normalisation buffers. Their frozen encoder is obtained from Torchvision's official ImageNet weights and checked with a full SHA-256 hash. The export verifies that all encoder tensors match those upstream weights before omitting them. Original full checkpoints remain unchanged locally.

The two-hour head improved aggregate positional teacher agreement, especially during carry, but did not improve yaw. The rest-height baseline still beats it on many stationary frames. The train/validation gap, poor box yaw and single tin validation clip make the dataset limitation visible. [Results](results.md#rgb-learning-experiment) includes both recipes, the baseline and their failure cases.

Cached RGB/mask inputs make prediction depth-free, but this is not a tested live raw-RGB system. SAM initialisation may have used depth proposals during dataset preparation. Neither online masking nor robot execution driven by the learned poses was evaluated. The saved RGB-D robot replay remains the main result; no further training or matching test recordings are planned.
