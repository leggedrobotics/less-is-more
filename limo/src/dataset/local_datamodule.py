from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from limo.src.dataset.limo_datset import (
    get_mission_dataset,
    parse_missions_csv,
)
from limo.src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LocalLimoDataModule(LightningDataModule):
    """LightningDataModule that reads a local Grand Tour zarr dataset.

    Identical interface to LimoDataModule but skips any HuggingFace pull.
    Use this after running dataset_builder to train on a locally built
    dataset without uploading to HuggingFace.

    dataset_folder should point to the directory that contains the
    grandtour/ subdirectory, i.e. the same value you would pass to
    LimoDataModule's dataset_folder.
    """

    def __init__(
        self,
        dataset_folder: str,
        missions_csv: str,
        dataset_type: Literal["tel", "geo", "aug"],
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_test: bool = False,
        with_side_cams: bool = False,
        image_size: Tuple[int, int] = (308, 476),
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset_folder = Path(dataset_folder) / "grandtour"
        self.missions_csv = Path(missions_csv)
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self.with_side_cams = with_side_cams
        self.image_size = image_size

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None or self.data_val is not None:
            return

        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )

        missions = parse_missions_csv(self.missions_csv)
        splits: dict[str, list[Dataset]] = defaultdict(list)

        _zarr_dirs = {
            "tel": ["teleop_paths"],
            "geo": ["geometric_paths"],
            "aug": ["teleop_paths", "geometric_paths"],
        }
        required = _zarr_dirs[self.dataset_type]

        for mission, split in missions.items():
            mission_dir = self.dataset_folder / mission
            if not mission_dir.exists():
                log.warning(
                    f"Mission directory not found locally, skipping: {mission_dir}"
                )
                continue
            missing = [d for d in required if not (mission_dir / "data" / d).exists()]
            if missing:
                needed = {"teleop_paths": "tel", "geometric_paths": "geo"}
                cmds = " && ".join(
                    f"uv run dataset_builder/src/build_paths.py dataset_type={needed[d]}"
                    for d in missing
                )
                raise FileNotFoundError(
                    f"Mission '{mission}' is missing zarr group(s) {missing} "
                    f"for dataset_type='{self.dataset_type}'. "
                    f"Run the dataset builder first: {cmds}"
                )
            splits[split].append(
                get_mission_dataset(
                    self.dataset_type,
                    self.dataset_folder,
                    mission,
                    transform,
                    self.with_side_cams,
                )
            )

        self.data_train = (
            ConcatDataset(splits["train"]) if splits.get("train") else None
        )
        self.data_val = ConcatDataset(splits["val"]) if splits.get("val") else None
        self.data_test = ConcatDataset(splits["test"]) if splits.get("test") else None

        for split_name, ds in [
            ("train", self.data_train),
            ("val", self.data_val),
            ("test", self.data_test),
        ]:
            if ds is not None:
                log.info(f"Split '{split_name}' has {len(ds)} samples")

    def train_dataloader(self) -> DataLoader[Any]:
        if self.data_train is None:
            raise RuntimeError("Train dataset not loaded. Call setup() first.")
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        if self.data_val is None:
            return DataLoader([])
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        if self.data_test is None:
            return DataLoader([])
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=self.shuffle_test,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
