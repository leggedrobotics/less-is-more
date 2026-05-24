# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**Less is More (LiMo)** is a transformer-based visual navigation policy for legged robots. Given a single RGB image and an SE(2) goal pose (x, y, yaw) in the robot-centric frame, it predicts a 50-waypoint SE(2) trajectory. Training data comes from the [Grand Tour dataset](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset).

The repo has two packages: **`limo`** (training and inference) and **`dataset_builder`** (MPPI-based path generation from Grand Tour missions). Both are installed via `uv sync`.

## Commands

```bash
# Install (base — limo only)
uv sync

# Install with dataset builder (Linux x86_64, Python 3.10 only; requires FastGeodis CUDA wheel)
uv sync --extra dataset_builder

# Quick sanity-check training run (5 epochs, 5% of data)
uv run limo/src/train.py experiment=train_limo_debug

# Full training
uv run limo/src/train.py experiment=train_limo_on_D_aug

# Build geo + tel paths for 3 example missions
uv run dataset_builder/src/build_paths.py --config-name build_example dataset_type=geo
uv run dataset_builder/src/build_paths.py --config-name build_example dataset_type=tel

# Quick training run on 3 example missions (no WandB needed)
uv run limo/src/train.py experiment=train_limo_local_example

# Training on all locally built missions (no WandB needed)
uv run limo/src/train.py experiment=train_limo_local

# Inference on example images
uv run limo/src/inference.py

# Lint
uv run ruff check .
```

Hydra overrides work from the CLI, e.g.:

```bash
uv run limo/src/train.py experiment=train_limo_debug data.batch_size=32 logger=null
```

## Architecture

### Model: `limo/src/models/`

`LimoNet` ([limo/src/models/components/limo_net.py](limo/src/models/components/limo_net.py)):

- **Backbone**: frozen DINOv2 ViT-S/14 (`dinov2_vits14`) — only LayerNorm layers are trainable. Loaded lazily via `setup()` to avoid torch.hub calls at import time.
- **Positional encoding**: learned 2D row + column embeddings added to patch tokens.
- **Goal conditioning**: SE(2) goal linearly projected to `embed_dim`, added to learned time embeddings to form `path_length` query vectors.
- **Decoder**: 4-layer Transformer decoder; queries are the time-stamped goal embeddings, keys/values are the image patch tokens.
- **Output**: linear projection → (B, 50, 3) SE(2) waypoints in robot base frame.

`LimoModel` ([limo/src/models/limo_model.py](limo/src/models/limo_model.py)) is the PyTorch Lightning wrapper. It handles train/val/test steps, WandB image logging (via `create_combined_visualization`), and checkpoint → SafeTensors conversion (via `SafetensorsCallback`).

A side-camera variant exists at [limo/src/models/components/limo_net_side_cams.py](limo/src/models/components/limo_net_side_cams.py).

### Data: `limo/src/dataset/`

`get_dataset()` in [limo/src/dataset/limo_datset.py](limo/src/dataset/limo_datset.py) is the main entry point. It:

1. Reads `missions_csv` (maps timestamp → train/val/test split).
2. Downloads only the required topics from HuggingFace (`leggedrobotics/grand_tour_dataset`) via `snapshot_download` + tar extraction.
3. Returns a dict of `ConcatDataset` keyed by split name.

Dataset types (`dataset_type` arg):

- `"tel"` — teleop paths only (`data/teleop_paths/` zarr group)
- `"geo"` — MPPI geometric paths only (`data/geometric_paths/` zarr group)
- `"aug"` — concatenation of both (used for best performance)

`LimoDataModule` ([limo/src/dataset/limo_datamodule.py](limo/src/dataset/limo_datamodule.py)) wraps `get_dataset()` for Lightning and handles HuggingFace downloads.

`LocalLimoDataModule` ([limo/src/dataset/local_datamodule.py](limo/src/dataset/local_datamodule.py)) reads from a locally built dataset produced by `dataset_builder` — no HuggingFace pull. Use with `experiment=train_limo_local` or `dataset=limo_local`.

### Dataset Builder: `dataset_builder/`

`build_paths.py` ([dataset_builder/src/build_paths.py](dataset_builder/src/build_paths.py)) is the entry point. It:

1. Downloads required zarr topics from HuggingFace via `utils/grandtour_hub.py`.
2. Iterates frames of each mission, runs the MPPI planner (D_geo) or extracts teleop paths (D_tel).
3. Writes zarr output under `data/dataset_builder/grandtour/<timestamp>/data/{geometric,teleop}_paths/`.

**MPPI planner** (`dataset_builder/mppi_planner/`):

- `MPPIPlanner` wraps `MPPIObjective` + `MPPIOptimizer`. ROS-free (PyTorch + numpy only).
- `MPPIObjective.set_observation()` takes a `GridMap2D` (robot-centric elevation grid), start SE(2), and goal SE(2). Computes traversability and a geodesic distance field (GDF) each call.
- Traversability is computed by `TraversabilityFilter` — a pretrained CNN whose weights live in `weights.npz`.
- GDF is computed via FastGeodis (`fast_geodis_wrapper.py`), routing around fatal-traversability cells.

**Data source**: `GrandTourZarrSource` ([dataset_builder/src/mission_data_source.py](dataset_builder/src/mission_data_source.py)) implements the `MissionDataSource` Protocol and reads the Grand Tour zarr layout. Provides elevation maps, DLIO odometry poses, and JPEG images. Swap in any Protocol-compatible class to use a different source (simulation, etc.).

**HuggingFace download**: `pull_mission_topics()` in [utils/grandtour_hub.py](utils/grandtour_hub.py) downloads specific zarr topics per mission from a given HF revision. Skips topics that already exist locally.

**Configs**: `dataset_builder/configs/build.yaml` (all 48 missions, 10 paths/frame) and `build_example.yaml` (3 missions, 1 path/frame). Config fields cover MPPI hyperparameters, goal sampling distributions, map size/resolution, `viz`, and `write_mode`.

### Zarr Dataset Layout (the canonical format)

```text
<mission_dir>/
├── data/
│   ├── teleop_paths/      # zarr group
│   │   ├── path           # (N, 50, 3) float32 — SE(2) waypoints, base frame
│   │   ├── goal           # (N, 3)     float32 — SE(2) goal, base frame
│   │   ├── image_id       # (N,)       int64   — index into images/hdr_*/
│   │   └── goal_time      # (N,)       float32
│   └── geometric_paths/   # same schema
└── images/
    ├── hdr_front/         <frame_id:06d>.jpeg
    ├── hdr_left/
    └── hdr_right/
```

Images are referenced by `image_id` (zero-padded 6-digit frame number). `MissionDataset.__getitem__` loads the JPEG, applies the transform (resize to 308×476, ToTensor), and returns `{image_front, goal, path}` (plus side cams if requested).

### Configuration: `limo/configs/`

Hydra with the structure: `train.yaml` (root) → `dataset/limo.yaml`, `model/limo.yaml`, `trainer/default.yaml`, `callbacks/default.yaml`. Experiment configs in `experiment/` override these. Paths are resolved dynamically in `_ensure_paths()` at the start of `train.py` and `inference.py` — `root_dir` is inferred from the script location, not from `os.getcwd()`.

## Key Conventions

- All paths and goals are in the **robot base frame** as SE(2): `(x, y, yaw)` in meters/radians.
- Standard image size: **308 × 476** (height × width). Patch grid: 22 × 34 at patch_size=14.
- Path length: **N=50** waypoints over a **T=5s** horizon.
- Weights are distributed as **SafeTensors** (not `.ckpt`). The `SafetensorsCallback` auto-exports after each Lightning checkpoint.
- Training uses WandB by default; pass `logger=null` or `logger=csv` to disable.
- `torch.set_float32_matmul_precision("medium")` is set globally in `train.py`.
