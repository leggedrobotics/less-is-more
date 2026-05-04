"""GrandTourMissionDataset: camera images and DLIO pose.

This dataset is the training-side view of one Grand Tour mission.
It does NOT compute elevation maps - use GrandTourZarrSource from
mission_data_source.py for the build pipeline.
"""

from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Shared helpers (also imported by mission_data_source.py)
# ---------------------------------------------------------------------------

def _quat_to_matrix(xyzw: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] -> 3x3 rotation matrix (numpy)."""
    x, y, z, w = xyzw
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _nearest_idx(timestamps: np.ndarray, t: float) -> int:
    """Return index of the nearest timestamp in a sorted array."""
    idx = np.searchsorted(timestamps, t)
    if idx == 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    before = timestamps[idx - 1]
    after = timestamps[idx]
    return idx - 1 if (t - before) <= (after - t) else idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GrandTourMissionDataset(Dataset):
    """Per-frame dataset for one Grand Tour mission.

    Each sample:
        image_front  : Tensor (3, H, W)
        pose_pos     : Tensor (3,)       - DLIO position in dlio_map frame
        pose_orien   : Tensor (4,)       - DLIO quaternion (x,y,z,w)
        timestamp    : float

    With with_side_cams=True also:
        image_left, image_right : Tensor (3, H, W)

    Required topics in mission_dir:
        data/hdr_front/          (zarr: timestamp)
        data/dlio_map_odometry/  (zarr: pose_pos, pose_orien, timestamp)
        images/hdr_front/        (JPEG files)
    """

    def __init__(
        self,
        mission_dir: Path | str,
        with_side_cams: bool = False,
        map_size: float = 4.0,
        map_resolution: float = 0.04,
        image_size: tuple[int, int] = (308, 476),
    ) -> None:
        self.mission_dir = Path(mission_dir)
        self.with_side_cams = with_side_cams
        self.map_size = map_size
        self.map_resolution = map_resolution

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self._z_cam = zarr.open_group(str(self.mission_dir / "data" / "hdr_front"), mode="r")
        self._z_dlio = zarr.open_group(str(self.mission_dir / "data" / "dlio_map_odometry"), mode="r")

        self._cam_ts = np.array(self._z_cam["timestamp"])
        self._dlio_ts = np.array(self._z_dlio["timestamp"])

        if with_side_cams:
            self._z_cam_left = zarr.open_group(str(self.mission_dir / "data" / "hdr_left"), mode="r")
            self._z_cam_right = zarr.open_group(str(self.mission_dir / "data" / "hdr_right"), mode="r")
            self._left_ts = np.array(self._z_cam_left["timestamp"])
            self._right_ts = np.array(self._z_cam_right["timestamp"])

    def __len__(self) -> int:
        return len(self._cam_ts)

    def __getitem__(self, idx: int) -> dict:
        timestamp = float(self._cam_ts[idx])

        img_path = self.mission_dir / "images" / "hdr_front" / f"{idx:06d}.jpeg"
        image_front = self.transform(Image.open(img_path).convert("RGB"))

        d_idx = _nearest_idx(self._dlio_ts, timestamp)
        pose_pos = torch.from_numpy(np.array(self._z_dlio["pose_pos"][d_idx], dtype=np.float32))
        pose_orien = torch.from_numpy(np.array(self._z_dlio["pose_orien"][d_idx], dtype=np.float32))

        sample = {
            "image_front": image_front,
            "pose_pos": pose_pos,
            "pose_orien": pose_orien,
            "timestamp": timestamp,
        }

        if self.with_side_cams:
            l_idx = _nearest_idx(self._left_ts, timestamp)
            r_idx = _nearest_idx(self._right_ts, timestamp)
            img_l = self.mission_dir / "images" / "hdr_left" / f"{l_idx:06d}.jpeg"
            img_r = self.mission_dir / "images" / "hdr_right" / f"{r_idx:06d}.jpeg"
            sample["image_left"] = self.transform(Image.open(img_l).convert("RGB"))
            sample["image_right"] = self.transform(Image.open(img_r).convert("RGB"))

        return sample
