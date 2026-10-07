# Third-party Software

No SAM source tree, simulator installation or downloaded foundation-model checkpoint is bundled in this repository.

| Component | Role | Upstream terms |
| --- | --- | --- |
| [SAM 2](https://github.com/facebookresearch/sam2) | Prompted video segmentation | [Apache-2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE) |
| [PyTorch](https://github.com/pytorch/pytorch) | Supervised learning | [BSD-style licence](https://github.com/pytorch/pytorch/blob/v2.7.0/LICENSE) |
| [Torchvision](https://github.com/pytorch/vision) | Frozen ResNet-18 encoder | [BSD-3-Clause](https://github.com/pytorch/vision/blob/v0.22.0/LICENSE) |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | Panda simulation and differential IK | [BSD-3-Clause](https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE) |
| [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/) | Rendering, physics and robot assets | NVIDIA's applicable software and asset terms |
| [Orbbec SDK](https://github.com/orbbec/OrbbecSDK_v2) | RGB-D capture | Upstream SDK and Python-binding terms |
| [OpenCV](https://github.com/opencv/opencv) | Images, calibration and ArUco detection | Upstream OpenCV and contributed-module licences |

ResNet-18 ImageNet weights are downloaded by Torchvision from its official URL. SAM 2.1 Tiny is downloaded separately from Meta. Their software licences should not be taken as a claim of ownership of the underlying training data. Keep upstream notices when installing or redistributing dependencies.

The downloaded Panda assets are referenced from Isaac Lab's configured asset location. They are not copied into the repository. The marker patterns in the scene are generated from OpenCV's `DICT_4X4_1000` dictionary.
