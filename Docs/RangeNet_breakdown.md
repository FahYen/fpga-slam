# RangeNet Paper Pipeline -> Folder Map

This note maps the RangeNet++ paper's four-stage pipeline to the code under `RangeNet/train`.

Paper reference flow:

- (A) point cloud -> range image
- (B) 2D fully-convolutional semantic segmentation
- (C) semantic transfer from 2D back to original 3D points
- (D) range-image-based 3D post-processing with fast GPU kNN

## A) Point Cloud -> Range Image

Primary folders:

- `common/`
- `tasks/semantic/dataset/`

What they do:

- `common/laserscan.py` performs spherical projection from XYZ points into range-image indices and channels.
- `tasks/semantic/dataset/kitti/parser.py` uses `LaserScan`/`SemLaserScan` to build model-ready projected tensors:
  - `proj_range`, `proj_xyz`, `proj_remission`, `proj_mask`
  - plus `proj_x`/`proj_y` index maps for later back-transfer.

## B) 2D Semantic Segmentation Network

- `tasks/semantic/modules/segmentator.py`: 
  - assembles backbone + decoder + segmentation head, applies softmax, and handles checkpoint load/save.
    - Current head is `Conv2d(..., kernel_size=3, padding=1)` after dropout.
    - Note: paper text describes a final `[1x1]` projection; implementation here uses `3x3`.
- `backbones/darknet.py`: 
  - encoder. Key paper-aligned changes are implemented here:
    - 5-channel input support (`range + x + y + z + remission`).
    - Horizontal-only downsampling (`stride=[1, 2]`) to preserve vertical LiDAR resolution.
    - Output-stride (`OS`) control and residual blocks for feature extraction.
- `tasks/semantic/decoders/darknet.py`: 
  - decoder with horizontal upsampling (`ConvTranspose2d` with `kernel=[1,4]`, `stride=[1,2]`) and skip-add fusion from encoder stages.
- `tasks/semantic/dataset/kitti/parser.py`: 
  - builds normalized 5-channel tensors consumed by the network (`proj = (proj - mean)/std`), so this is part of the effective network input contract.
- `tasks/semantic/modules/trainer.py`: 
  - training-time objective and optimization details for this network block:
    - class-frequency-weighted cross-entropy (`NLLLoss` over `log(softmax)` outputs),
    - SGD optimizer and LR schedule.
- `tasks/semantic/modules/user.py`: 
  - inference path (softmax -> argmax per pixel) and optional handoff to KNN post-processing.
- `tasks/semantic/config/arch/*.yaml`: 
  - controls architecture and interface-critical parameters (enabled input channels, `OS`, dropout/BN, image width/height, normalization means/stds, optional post-processing flags).

## C) 2D -> 3D Semantic Transfer

Primary folders:

- `tasks/semantic/modules/`
- `tasks/semantic/dataset/`

What they do:

- Inference produces per-pixel class predictions on the projected image.
- Predictions are transferred back to original point order by indexing with `proj_y`/`proj_x`.
- This recovers labels for all original points independent of projection discretization.

Key path:

- `tasks/semantic/modules/user.py` (`unproj_argmax = proj_argmax[p_y, p_x]` when KNN is disabled).

## D) 3D Post-Processing (GPU kNN)

Primary folder:

- `tasks/semantic/postproc/`

What it does:

- `postproc/KNN.py` applies a fast range-image neighborhood kNN refinement.
- It gathers local neighbors around each projected point, applies range/distance weighting, and votes a refined label per original 3D point.
- Implemented as PyTorch ops and runs on GPU when tensors are on CUDA.

## Orchestration and Utilities Around A-D

Entry points:

- `tasks/semantic/train.py` + `tasks/semantic/modules/trainer.py`: training pipeline.
- `tasks/semantic/infer.py` + `tasks/semantic/modules/user.py`: inference pipeline.
- `tasks/semantic/export_sgslam_labels.py`: batch exporter used for `RangeNet -> .label -> SG-SLAM`.

Supporting config/data:

- `tasks/semantic/config/labels/`: dataset label definitions and mappings (`learning_map`, `learning_map_inv`).
- `common/` helpers (meters, schedulers, logging, visualization) support runs but are not separate paper stages.

Out of scope for this semantic A-D mapping:

- `tasks/panoptic/` (different task family).

