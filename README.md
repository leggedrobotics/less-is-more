<h1 align="center">Less is More 🍋: Scalable Visual Navigation<br>from Limited Data</h1>

<p align="center">
📄 <a href="https://arxiv.org/pdf/2601.17815">Paper</a> | 🌐 <a href="https://leggedrobotics.github.io/less-is-more/">Project Page</a> | 🤗 <a href="https://huggingface.co/yv1es/less-is-more">Weights</a> | 📊 <a href="https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset">Dataset</a>
</p>

**Less is More (LiMo)** is a transformer-based visual navigation policy that predicts goal-conditioned SE(2) trajectories from a single RGB observation. We demonstrate that augmenting limited expert demonstrations with geometric planner-generated trajectories yields substantial performance improvements, achieving robust visual navigation through strategic data curation rather than simply collecting more data.

## Release Status

- [x] Inference code
- [x] Checkpoints released (SafeTensors on HuggingFace)
- [x] Dataset and training code
- [x] Dataset builder with MPPI planner
- [ ] ROS integration

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. We assume an NVIDIA GPU with driver version ≥530 (for CUDA 12.1 support).

```bash
git clone https://github.com/leggedrobotics/less-is-more \
  && cd less-is-more \
  && uv sync \
  && uv pip install -e .
```

The codebase uses [Hydra](https://hydra.cc/) for configuration management in the `limo` package.

### Download Pretrained Weights

Download the pretrained LiMo checkpoint:

```bash
wget -O data/weights/limo_trained_on_D_aug.safetensors \
  https://huggingface.co/yv1es/less-is-more/resolve/main/limo_trained_on_D_aug.safetensors
```

## Quick Start

Run inference on the provided examples:

```bash
uv run limo/src/inference.py
```

The default configuration in `limo/configs/inference.yaml` processes images from the example dataset in `data/inference_example/`.

<div align="center">
  <img src="data/inference_example/output/eth_indoor_output.png" alt="ETH Indoor Navigation" width="100%">
  <p><i>Example: Inside ETH Zurich main building with multiple predictions</i></p>
</div>

<div align="center">
  <img src="data/inference_example/output/rocks_output.png" alt="Rocky Terrain Navigation" width="100%">
  <p><i>Example: Outdoor mountain trail with multiple predictions</i></p>
</div>

### Configuration

The inference pipeline is configured via `limo/configs/inference.yaml`:

- `weights_path`: Path to model weights (default: `data/weights/limo_trained_on_D_aug.safetensors`)
- `input_path`: Single image file or directory containing `.jpg`, `.png`, or `.jpeg` files
- `goals_csv`: CSV file defining navigation targets (format: `x,y,yaw` in meters/radians, robot frame)
- `camera_info`: Optional YAML file with camera intrinsics for path projection visualization
- `output_dir`: Directory for generated visualizations

See `data/inference_example/` for a complete working example with sample images, goals, and camera calibration.

## Training

The training pipeline uses [PyTorch Lightning](https://lightning.ai/) for model training and [Hydra](https://hydra.cc/) for configuration management.

### Training Script

Train the model using: :

```bash
uv run limo/src/train.py experiment=train_limo_debug
```

(this debug experiment config will only start a quick debug run to test your system)

The main entry point is `limo/src/train.py`, which:

- Loads configuration from `limo/configs/train.yaml` and an experiment config
- Automatically pulls the dataset from HuggingFace
- Runs training with logging
- Saves checkpoints and weights

### Configuration System

The training pipeline uses Hydra with the following structure:

**Config directories** (`limo/configs/`):

- `data/`: Dataset configuration (e.g., `limo.yaml` - LimoDataModule settings)
- `model/`: Model architecture (e.g., `limo.yaml` - network configuration)
- `trainer/`: PyTorch Lightning trainer settings
- `logger/`: Logging backends (e.g., wandb, csv)
- `callbacks/`: Training callbacks (e.g., checkpointing)
- `experiment/`: Complete experiment configs that override defaults

**Experiment configs** (`limo/configs/experiment/`):

- `train_limo_debug.yaml`: Quick debug run with 5 epochs, limited batches
- `train_limo_on_D_aug.yaml`: Full training on augmented dataset
- `train_limo_side_cams.yaml`: Training with side camera inputs

Override configs using the Hydra syntax:

```bash
# Use a specific experiment
uv run limo/src/train.py experiment=train_limo_on_D_aug

# Override specific parameters
uv run limo/src/train.py experiment=train_limo_debug data.batch_size=32 trainer.max_epochs=10

# Disable wandb logging
uv run limo/src/train.py experiment=train_limo_debug logger=null
```

By default, training uses [Weights & Biases](https://wandb.ai/) for logging.

### Checkpoints and Weights

- **Checkpoints**: Saved to `logs/train/runs/<timestamp>/checkpoints/` (PyTorch Lightning format)
- **SafeTensors weights**: Automatically converted and saved to `logs/train/runs/<timestamp>/weights/` after each checkpoint

## Dataset Builder

The `dataset_builder` package generates training samples from Grand Tour missions. It runs the MPPI geometric planner over pre-computed elevation maps to produce goal-conditioned paths, writing output in the same zarr format consumed by the training pipeline. Required topics are downloaded from HuggingFace automatically.

### Quick start: build and train on 3 missions

```bash
# Build D_geo paths for the three ETH missions (1 path/frame, ~10 min)
uv run dataset_builder/src/build_paths.py --config-name build_example dataset_type=geo

# Train on the result
uv run limo/src/train.py dataset=limo_local
```

Output goes to `data/dataset_builder/`. The `limo_local` dataset config points there and uses `missions_split_example.csv` (the same three missions, split into train/val/test).

### Full build (all missions, as in the paper)

```bash
# D_geo: 10 paths per frame across all 48 missions
uv run dataset_builder/src/build_paths.py dataset_type=geo

# D_tel: teleoperation paths
uv run dataset_builder/src/build_paths.py dataset_type=tel
```

To train on all missions, override the missions CSV:

```bash
uv run limo/src/train.py dataset=limo_local \
  data.missions_csv=limo/configs/dataset/missions_split.csv
```

### Visualization

Pass `viz=true` to watch the planner and maps while building:

```bash
uv run dataset_builder/src/build_paths.py --config-name build_example dataset_type=geo viz=true viz_every=50
```

## Dataset

LiMo's training data is based on **Grand Tour dataset** from HuggingFace: [`leggedrobotics/grand_tour_dataset`](https://huggingface.co/datasets/leggedrobotics/grand_tour_dataset)
We added LiMo's data to the same HuggingFace repo.

> **Note**: The dataset is **automatically pulled** from HuggingFace. No manual download required!

### Dataset Types

The Grand Tour dataset contains multiple mission recordings. Different sample types can be extracted:

- **Teleoperation samples (`tel`)**: Expert demonstrations from human teleoperation
- **Geometric samples (`geo`)**: Trajectories generated by the MPPI geometric planner
- **Augmented samples (`aug`)**: Combined set of teleoperation + geometric samples

Select the dataset type using the `get_dataset()` method in `dataset/src/limo_datset.py`:

```python
from dataset.src.limo_datset import get_dataset

# Load teleoperation samples only
dataset_tel = get_dataset(
    dataset_type="tel",           # or "geo", "aug"
    dataset_folder="data/dataset",
    missions_csv="missions_split.csv",
    with_side_cams=False
)

# Load augmented samples with side cameras
dataset_aug = get_dataset(
    dataset_type="aug",
    dataset_folder="data/dataset",
    missions_csv="missions_split.csv",
    with_side_cams=True
)
```

### Missions and Train/Val/Test Split

The `missions_split.csv` file controls which missions are used and how they're split:

**CSV Format**:

```
Mission,Timestamp,Split
grandtour_mission_1,2024-01-15,train
grandtour_mission_2,2024-01-16,val
grandtour_mission_3,2024-01-17,test
```

**Usage**:

- Include only specific missions by adding rows to the CSV
- Control split ratios by adjusting the number of missions per split
- Quick debugging: Use a subset of missions (see `limo/configs/data/missions_split_debug.csv`)

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{inglin2026morescalablevisualnavigation,
      title={Less Is More: Scalable Visual Navigation from Limited Data},
      author={Yves Inglin and Jonas Frey and Changan Chen and Marco Hutter},
      year={2026},
      eprint={2601.17815},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.17815},
}
```
