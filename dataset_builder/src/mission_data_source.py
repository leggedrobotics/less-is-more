"""MissionDataSource protocol + GrandTourZarrSource implementation.

MissionDataSource defines the interface consumed by the path-building pipeline
(build_paths.py). Swap in any other class that satisfies the protocol to feed
geometry/images from a different source (simulation, custom bags, etc.).

GrandTourZarrSource reads the Grand Tour mission zarr layout:
    data/hdr_front/         - camera timestamps
    data/dlio_map_odometry/ - robot trajectory (world frame)
    data/elevation_map/     - pre-computed robot-centric elevation (200x200, 4 cm)
    images/hdr_front/       - JPEG images
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import zarr
from PIL import Image

from dataset_builder.src.grand_tour_dataset import _nearest_idx, _quat_to_matrix


class MissionDataSource(Protocol):
    map_size: float
    map_resolution: float

    def __len__(self) -> int: ...

    def get_timestamp(self, i: int) -> float: ...

    def get_image(self, i: int) -> np.ndarray:
        """Return (H, W, 3) uint8 RGB."""
        ...

    def get_elevation(self, i: int) -> np.ndarray:
        """Return (N, N) float32 elevation, NaN = unknown, robot-centric yaw-aligned."""
        ...

    def get_pose_se2_world(self, i: int) -> np.ndarray:
        """Return (3,) [x, y, yaw] in the world (dlio_map) frame."""
        ...

    def get_trajectory_world(self, i: int, duration: float, n: int) -> np.ndarray:
        """Return (n, 3) SE2 waypoints in world frame starting at frame i's timestamp.

        Samples n evenly-spaced waypoints over [t_i, t_i + duration].
        Used by the D_tel builder.
        """
        ...


class GrandTourZarrSource:
    """Reads one Grand Tour mission zarr for the build pipeline."""

    def __init__(
        self,
        mission_dir: Path | str,
        map_size: float = 4.0,
        map_resolution: float = 0.04,
    ) -> None:
        self.mission_dir = Path(mission_dir)
        self.map_size = map_size
        self.map_resolution = map_resolution

        self._z_cam = zarr.open_group(str(self.mission_dir / "data" / "hdr_front"), mode="r")
        self._z_dlio = zarr.open_group(str(self.mission_dir / "data" / "dlio_map_odometry"), mode="r")
        self._z_elev = zarr.open_group(str(self.mission_dir / "data" / "elevation_map"), mode="r")

        self._cam_ts = np.array(self._z_cam["timestamp"])
        self._dlio_ts = np.array(self._z_dlio["timestamp"])
        self._dlio_pos = np.array(self._z_dlio["pose_pos"])      # (M, 3)
        self._dlio_orien = np.array(self._z_dlio["pose_orien"])  # (M, 4) xyzw

    def __len__(self) -> int:
        return len(self._cam_ts)

    def get_timestamp(self, i: int) -> float:
        return float(self._cam_ts[i])

    def get_image(self, i: int) -> np.ndarray:
        img_path = self.mission_dir / "images" / "hdr_front" / f"{i:06d}.jpeg"
        return np.array(Image.open(img_path).convert("RGB"))

    def get_elevation(self, i: int) -> np.ndarray:
        return np.array(self._z_elev["elevation"][i], dtype=np.float32)

    def get_pose_se2_world(self, i: int) -> np.ndarray:
        ts = self.get_timestamp(i)
        d_idx = _nearest_idx(self._dlio_ts, ts)
        pos = self._dlio_pos[d_idx]
        orien = self._dlio_orien[d_idx]  # xyzw
        R = _quat_to_matrix(orien)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return np.array([pos[0], pos[1], yaw], dtype=np.float32)

    def get_trajectory_world(self, i: int, duration: float, n: int) -> np.ndarray:
        """Interpolate DLIO poses over [t_i, t_i + duration] at n equally-spaced steps."""
        t0 = self.get_timestamp(i)
        times = np.linspace(t0, t0 + duration, n)
        waypoints = np.zeros((n, 3), dtype=np.float32)
        for k, t in enumerate(times):
            idx = _nearest_idx(self._dlio_ts, t)
            pos = self._dlio_pos[idx]
            orien = self._dlio_orien[idx]
            R = _quat_to_matrix(orien)
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            waypoints[k] = [pos[0], pos[1], yaw]
        return waypoints
