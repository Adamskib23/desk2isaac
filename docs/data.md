# Data and Reproducibility

## Collection

I personally recorded overhead desk manipulation with an Orbbec Gemini 2. Four flat ArUco markers (IDs 0-3, `DICT_4X4_1000`, 106.4 mm black squares) define the calibrated workspace. Each retained clip contains one object's pick-and-place motion, with stationary start and end windows.

| Object | Measured dimensions | Simulation shape |
| --- | --- | --- |
| Black box | 55.55 x 45 x 31 mm | Cuboid |
| Blue sphere | 36.5 mm diameter | Sphere |
| Yellow mustard tin | 51 x 35 x 85 mm | Cuboid approximation |

A dark shot glass was tried early, then replaced because tracking and depth were unreliable. Its historical configuration is not part of the evaluated three-object dataset.

The retained set has 30 clips, ten per object, with 8,969 RGB-D frames. Fifty frames were gated during tracking preparation. Failed recordings and preliminary colour-mask attempts were not included in the retained set. Robot review outcome 18, the sphere failure, remains in the results.

## What Is Included

| Location | Contents | What it reproduces |
| --- | --- | --- |
| `examples/retargeting/` | Prepared box, sphere and tin tasks (review indices 23, 28, 30); numeric marker/background snapshots | Physical robot replay from personally collected motion |
| `examples/rgb_supervision/` | All 3,007 RGB crops, geometry features, calibration snapshots and separate teacher labels | Both supervised fitting recipes and validation prediction |
| `examples/models/` | Short- and long-run compact learned heads | Prediction without fitting again |
| `evidence/robot/` | All 30 review outcomes, per-clip physical results and summary | Inspect the reported success rate and failure |
| `evidence/rgb_short/`, `evidence/rgb_long/` | Reports, histories, settings, predictions and validation metrics | Inspect checkpoint selection and teacher disagreement |
| `docs/assets/` | Existing screenshot, three paired videos, validation plots and loss curves | Visual inspection of the demonstrated behaviour |

The videos come from the recorded review, not a newly staged or generated example. The compact task scene snapshots retain the original numeric geometry but are bound to the packaged tasks with relative paths. They do not rerun raw-depth calibration. Source hashes in [provenance](../evidence/provenance.json) identify the originals; the submission file manifest identifies the public copies.

## What Is Not Included

Full-resolution raw RGB-D recordings, per-frame SAM masks, intermediate videos, failed captures, downloaded foundation models, virtual environments, simulator assets and development logs are omitted. They are preserved locally rather than deleted. Full 30-episode raw-data regeneration is therefore not possible from this compact repository alone. Three prepared physics tasks and the complete cropped supervised dataset are runnable.

Source-episode strings and hashes in evidence identify provenance; they are not promises that every historical `data/real/` or `outputs/` path exists after cloning.

## Supervised Split

| Split | Clips | Sampled frames | Objects |
| --- | --- | --- | --- |
| Train | 22 | 2,202 | 5 box, 8 sphere, 9 tin clips |
| Validation | 8 | 805 | 5 box, 2 sphere, 1 tin clips |
| Independent test | 0 | 0 | Not recorded |

Sampling stride is three frames. Clips from batch 003 (indices 23-30) form the validation set. Nested recording-batch metadata is used to keep a batch together; all derivatives of a recording stay on the same side. Feature normalisation is fitted to unaugmented training views only. Validation images are not augmented.

Validation selected the best checkpoint repeatedly, especially in the two-hour run. Do not call its scores unbiased test performance. Recording another test set was considered, but the marker layout subsequently moved; the current experiment is frozen without one.

## Inputs and Targets

`inputs.json` contains RGB crop paths and hashes, 15 RGB/mask/known-object geometry features, calibration snapshots and split metadata. Frame and clip identifiers join records but are not fed to the network. `teacher_labels.json` contains the approximate observed centre/yaw targets and motion phases. `training_manifest.json` binds both files with hashes.

Teacher labels are pseudo-labels from the depth/geometry tracker, not manually surveyed ground truth. Only accepted observed candidates are sampled. Interpolated guidance and predicted poses are not promoted to observations. The manifest keeps `training_ready: false`; the training command requires `--allow-pseudo-labels` to acknowledge this limitation.

The public copy normalises absolute machine paths in metadata. Crop bytes, numeric features, calibration, labels, grouping and split assignment are unchanged. Its input-file hash consequently differs from the historical hash stored in the original checkpoint; both hashes are recorded. The compact checkpoint removes only the frozen ImageNet encoder after checking it against the upstream weights.

Prediction requires no depth or teacher files, but does require the prepared mask-derived inputs and saved calibration. These prerequisites matter when interpreting "RGB-only".

Code and recorded-data permissions are documented in [MIT licence](../LICENSE), [data terms](../DATA_TERMS.md) and [third-party terms](../THIRD_PARTY.md).
