import csv
import re
import shutil
import tarfile
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Literal, Tuple

import torch
import zarr
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

from limo.src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def parse_missions_csv(missions_csv: Path) -> dict[str, str]:
    """Parse missions CSV and return a dict mapping Timestamp to Split."""
    timestamp_to_split = {}
    with missions_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            timestamp = row.get("Timestamp", "").strip()
            split = row.get("Split", "").strip()
            if timestamp:
                timestamp_to_split[timestamp] = split
    return timestamp_to_split


def pull_missions_from_hf(
    missions: list[str], topics: list[str], dataset_folder: Path
) -> Path:
    allow_patterns = []
    for mission, topic in product(missions, topics):
        allow_patterns.append(f"{mission}/*{topic}*")

    log.info("Downloading missions from Hugging Face...")
    hf_data_cache = snapshot_download(
        repo_id="leggedrobotics/grand_tour_dataset",
        revision="refs/pr/6",  # REMOVE LATER
        allow_patterns=allow_patterns,
        repo_type="dataset",
    )

    log.info(f"Extraction missions from HF cache at {hf_data_cache}...")
    move_dataset(hf_data_cache, dataset_folder, allow_patterns=allow_patterns)
    return Path(dataset_folder)


def move_dataset(cache, dataset_folder, allow_patterns=["*"]):
    def convert_glob_patterns_to_regex(glob_patterns):
        regex_parts = []
        for pat in glob_patterns:
            # Escape regex special characters except for * and ?
            pat = re.escape(pat)
            # Convert escaped glob wildcards to regex equivalents
            pat = pat.replace(r"\*", ".*").replace(r"\?", ".")
            # Make sure it matches full paths
            regex_parts.append(f".*{pat}$")

        # Join with |
        combined = "|".join(regex_parts)
        return re.compile(combined)

    pattern = convert_glob_patterns_to_regex(allow_patterns)
    files = [f for f in Path(cache).rglob("*") if pattern.match(str(f))]
    tar_files = [f for f in files if f.suffix == ".tar"]

    for source_path in tar_files:
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(source_path, "r") as tar:
                tar.extractall(path=dest_path.parent)
        except tarfile.ReadError as e:
            log.error(f"Error opening or extracting tar file '{source_path}': {e}")
        except Exception as e:
            log.error(
                f"An unexpected error occurred while processing {source_path}: {e}"
            )

    other_files = [f for f in files if not f.suffix == ".tar" and f.is_file()]
    for source_path in other_files:
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)


class MissionDataset(Dataset):
    def __init__(
        self,
        dataset_type: Literal["tel", "geo", "aug"],
        dataset_folder: Path,
        mission_name: str,
        transform: transforms.Compose,
        with_side_cams: bool = False,
    ):
        self.dataset_type = dataset_type
        self.dataset_folder = dataset_folder
        self.mission_name = mission_name
        self.transform = transform
        self.with_side_cams = with_side_cams

        mission_dir = dataset_folder / mission_name
        if not mission_dir.exists():
            err = f"Mission dataset '{mission_name}' not found in {dataset_folder}"
            log.error(err)
            raise FileNotFoundError(err)

        if dataset_type == "tel":
            self.z = zarr.open_group(
                str(mission_dir / "data" / "teleop_paths"), mode="r"
            )
        elif dataset_type == "geo":
            self.z = zarr.open_group(
                str(mission_dir / "data" / "geometric_paths"), mode="r"
            )
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

    def __len__(self):
        return len(self.z["path"])

    def load_image(self, topic: str, idx: int) -> Image.Image:
        image_path = (
            self.dataset_folder
            / self.mission_name
            / "images"
            / topic
            / f"{self.z['image_id'][idx]:06d}.jpeg"
        )
        if not image_path.exists():
            log.error(f"Image not found at {image_path}")
            raise FileNotFoundError(f"Image not found at {image_path}")
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, idx):
        image_front = self.load_image("hdr_front", idx)
        image_front = self.transform(image_front)

        goal = torch.tensor(self.z["goal"][idx], dtype=torch.float32)
        path = torch.tensor(self.z["path"][idx], dtype=torch.float32)

        batch = {
            "image_front": image_front,
            "goal": goal,
            "path": path,
        }

        if self.with_side_cams:
            image_left = self.load_image("hdr_left", idx)
            image_left = self.transform(image_left)
            batch["image_left"] = image_left

            image_right = self.load_image("hdr_right", idx)
            image_right = self.transform(image_right)
            batch["image_right"] = image_right

        return batch


def get_mission_dataset(
    dataset_type: Literal["tel", "geo", "aug"],
    dataset_folder: Path,
    mission_name: str,
    transform: transforms.Compose,
    with_side_cams: bool = False,
) -> Dataset:
    if dataset_type == "aug":
        geo_ds = MissionDataset(
            "geo", dataset_folder, mission_name, transform, with_side_cams
        )
        tel_ds = MissionDataset(
            "tel", dataset_folder, mission_name, transform, with_side_cams
        )
        return ConcatDataset([geo_ds, tel_ds])
    if dataset_type in ["tel", "geo"]:
        return MissionDataset(
            dataset_type, dataset_folder, mission_name, transform, with_side_cams
        )
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")


def get_dataset(
    dataset_type: Literal["tel", "geo", "aug"],
    dataset_folder: Path,
    missions_csv: Path,
    with_side_cams: bool = False,
    image_size: Tuple[int, int] = (308, 476),
):
    missions = parse_missions_csv(missions_csv)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )

    topics = ["hdr_front"]
    if with_side_cams:
        topics += ["hdr_left", "hdr_right"]
    if dataset_type in ["tel", "aug"]:
        topics.append("teleop_paths")
    if dataset_type in ["geo", "aug"]:
        topics.append("geometric_paths")

    grandtour_folder = dataset_folder / "grandtour"
    grandtour_folder.mkdir(parents=True, exist_ok=True)
    datset_dir = pull_missions_from_hf(list(missions.keys()), topics, grandtour_folder)

    datasets = defaultdict(list)
    for mission, split in missions.items():
        datasets[split].append(
            get_mission_dataset(
                dataset_type, datset_dir, mission, transform, with_side_cams
            )
        )

    splits: dict[str, Dataset] = dict()
    for split, ds_list in datasets.items():
        splits[split] = ConcatDataset(ds_list)
        log.info(f"Split '{split}' has {len(splits[split])} samples")
    return splits
