import csv
from pathlib import Path
from typing import List, Optional

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from PIL import Image
from safetensors.torch import load_file

from limo.src.models.components.limo_net import LimoNet
from limo.src.utils import RankedLogger
from limo.src.utils.visualization import (
    create_combined_visualization,
    load_image_and_normalize,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _ensure_paths(cfg: DictConfig) -> None:
    if not cfg.get("paths"):
        return
    if not cfg.paths.get("root_dir"):
        root_dir = Path(__file__).resolve().parents[2]
        cfg.paths.root_dir = str(root_dir)
    if not cfg.paths.get("data_dir"):
        cfg.paths.data_dir = str(Path(cfg.paths.root_dir) / "data")
    if not cfg.paths.get("log_dir"):
        cfg.paths.log_dir = str(Path(cfg.paths.root_dir) / "logs")


def load_model(cfg: DictConfig) -> torch.nn.Module:
    """Load torch model from SafeTensors weights."""
    weights_path = Path(cfg.weights_path)
    log.info(f"Loading model from weights: {weights_path}")
    model: torch.nn.Module = LimoNet(pretrained=False)
    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_goals_from_csv(csv_path: str) -> np.ndarray:
    """Load goals from CSV file. Expected columns: x, y, yaw (in robot-centric frame)."""
    goals = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row["x"])
            y = float(row["y"])
            yaw = float(row["yaw"])
            goals.append([x, y, yaw])
    goals = np.array(goals)
    log.info(f"Loaded {len(goals)} goals from {csv_path}")
    return goals


def load_camera_info(yaml_path: str) -> Optional[dict]:
    """Load camera calibration from YAML file."""
    with open(yaml_path, "r") as f:
        camera_info = yaml.safe_load(f)
    log.info(f"Loaded camera info from {yaml_path}")
    return camera_info


def process_single_image(
    image_path: str,
    model: torch.nn.Module,
    goals: np.ndarray,
    camera_info: Optional[dict],
) -> np.ndarray:
    """
    Process a single image and create combined visualization.

    Args:
        image_path: Path to input image
        model: PyTorch model for prediction
        goals: (M, 3) array of goals (x, y, yaw)
        camera_info: Dict with camera calibration, or None

    Returns:
        combined_image: Visualization with camera view left, BEV right
    """
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)

    image_tensor = load_image_and_normalize(image_path)

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    predicted_paths = []
    with torch.no_grad():
        for goal in goals:
            goal_tensor = (
                torch.tensor(goal, dtype=torch.float32).unsqueeze(0).to(device)
            )
            pred_path = model({"image_front": image_tensor, "goal": goal_tensor})
            pred_path = pred_path.cpu().numpy()[0]
            predicted_paths.append(pred_path)

    combined_image = create_combined_visualization(
        original_image, predicted_paths, goals, camera_info
    )

    return combined_image


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    _ensure_paths(cfg)

    if cfg.weights_path is None:
        raise ValueError("weights_path must be specified")
    if cfg.input_path is None:
        raise ValueError("input_path must be specified")
    if cfg.goals_csv is None:
        raise ValueError("goals_csv must be specified")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    goals = load_goals_from_csv(cfg.goals_csv)
    if len(goals) == 0:
        log.warning(
            "No goals found in CSV file. Inference will not produce any outputs."
        )

    camera_info = None
    if cfg.camera_info is not None:
        camera_info = load_camera_info(cfg.camera_info)

    input_path = Path(cfg.input_path)

    if input_path.is_file():
        output_image = process_single_image(str(input_path), model, goals, camera_info)
        output_path = output_dir / f"{input_path.stem}_output.png"
        Image.fromarray(output_image).save(output_path)
        log.info(f"Saved output to: {output_path}")

    elif input_path.is_dir():
        image_files = sorted(
            list(input_path.glob("*.jpg"))
            + list(input_path.glob("*.png"))
            + list(input_path.glob("*.jpeg"))
        )

        if len(image_files) == 0:
            log.warning(f"No image files (.jpg, .png, .jpeg) found in {input_path}")

        for img_path in image_files:
            log.info(f"Processing image: {img_path.name}")
            output_image = process_single_image(
                str(img_path), model, goals, camera_info
            )
            output_path = output_dir / f"{img_path.stem}_output.png"
            Image.fromarray(output_image).save(output_path)
            log.info(f"Saved output to: {output_path}")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    log.info("Inference complete!")


if __name__ == "__main__":
    main()
