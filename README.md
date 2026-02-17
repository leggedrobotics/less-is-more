<h1 align="center">Less is More 🍋: Scalable Visual Navigation<br>from Limited Data</h1>

<p align="center">
📄 <a href="https://arxiv.org/pdf/2601.17815">Paper</a> | 🌐 <a href="https://leggedrobotics.github.io/less-is-more/">Project Page</a>
</p>

**Less is More (LIMO)** is a transformer-based visual navigation policy that predicts goal-conditioned SE(2) trajectories from a single RGB observation. We demonstrate that augmenting limited expert demonstrations with geometric planner-generated trajectories yields substantial performance improvements, achieving robust visual navigation through strategic data curation rather than simply collecting more data.

## Release Status

- [x] Inference code and checkpoints
- [ ] Dataset and training code
- [ ] Dataset builder with MPPI planner

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. We assume an NVIDIA GPU with driver version ≥530 (for CUDA 12.1 support).

```bash
uv sync
uv pip install -e .
```

The codebase uses [Hydra](https://hydra.cc/) for configuration management in the `limo` package.

## Quick Start

Run inference on the provided examples:

```bash
uv run limo/src/inference.py
```

The default configuration in `limo/configs/inference.yaml` processes images from the example dataset in `data/inference_example/`.

<div align="center">
  <img src="data/inference_example/output/eth_indoor_output.png" alt="ETH Indoor Navigation" width="100%">
  <p><i>Example: Indoor corridor navigation with multiple goal predictions</i></p>
</div>

<div align="center">
  <img src="data/inference_example/output/rocks_output.png" alt="Rocky Terrain Navigation" width="100%">
  <p><i>Example: Outdoor rocky terrain navigation with path planning</i></p>
</div>

### Configuration

The inference pipeline is configured via `limo/configs/inference.yaml`:

- `ckpt_path`: Path to model weights (default: `data/weights/limo_trained_on_D_aug.safetensors`)
- `input_path`: Single image file or directory containing `.jpg`, `.png`, or `.jpeg` files
- `goals_csv`: CSV file defining navigation targets (format: `x,y,yaw` in meters/radians, robot frame)
- `camera_info`: Optional YAML file with camera intrinsics for path projection visualization
- `output_dir`: Directory for generated visualizations

See `data/inference_example/` for a complete working example with sample images, goals, and camera calibration.

> **Note**: Model weights will be released on HuggingFace soon.

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

## Contact

For questions or issues, please open an issue on GitHub.
