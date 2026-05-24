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


def _quat_to_matrix(xyzw: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] -> 3x3 rotation matrix (numpy)."""
    x, y, z, w = xyzw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _nearest_idx(timestamps: np.ndarray, t: float) -> int:
    """Return index of the nearest timestamp in a sorted array."""
    idx = np.searchsorted(timestamps, t)
    if idx == 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    return idx - 1 if (t - timestamps[idx - 1]) <= (timestamps[idx] - t) else idx


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

        self._z_cam = zarr.open_group(
            str(self.mission_dir / "data" / "hdr_front"), mode="r"
        )
        self._z_dlio = zarr.open_group(
            str(self.mission_dir / "data" / "dlio_map_odometry"), mode="r"
        )
        self._z_elev = zarr.open_group(
            str(self.mission_dir / "data" / "elevation_map"), mode="r"
        )

        self._cam_ts = np.array(self._z_cam["timestamp"])
        self._dlio_ts = np.array(self._z_dlio["timestamp"])
        self._dlio_pos = np.array(self._z_dlio["pose_pos"])  # (M, 3)
        self._dlio_orien = np.array(self._z_dlio["pose_orien"])  # (M, 4) xyzw

        self._side_ts: dict[str, np.ndarray | None] = {}
        for _cam in ("hdr_left", "hdr_right"):
            _zp = self.mission_dir / "data" / _cam
            if _zp.exists():
                self._side_ts[_cam] = np.array(
                    zarr.open_group(str(_zp), mode="r")["timestamp"]
                )
            else:
                self._side_ts[_cam] = None

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
        # DLIO body frame is rotated +90° (CCW) relative to ANYmal base_link;
        # subtract π/2 to recover the true base heading.
        yaw = float(np.arctan2(R[1, 0], R[0, 0])) - np.pi / 2
        return np.array([pos[0], pos[1], yaw], dtype=np.float32)

    def get_trajectory_world(self, i: int, duration: float, n: int) -> np.ndarray:
        """Sample DLIO poses over [t_i, t_i + duration] at n equally-spaced steps (nearest-neighbour)."""
        t0 = self.get_timestamp(i)
        times = np.linspace(t0, t0 + duration, n)
        idxs = np.searchsorted(self._dlio_ts, times).clip(0, len(self._dlio_ts) - 1)
        prev = (idxs - 1).clip(0)
        use_prev = np.abs(self._dlio_ts[prev] - times) < np.abs(
            self._dlio_ts[idxs] - times
        )
        idxs = np.where(use_prev, prev, idxs)
        pos = self._dlio_pos[idxs]  # (n, 3)
        orien = self._dlio_orien[idxs]  # (n, 4) xyzw
        x, y, z, w = orien[:, 0], orien[:, 1], orien[:, 2], orien[:, 3]
        yaw = np.arctan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z)) - np.pi / 2
        return np.column_stack([pos[:, 0], pos[:, 1], yaw]).astype(np.float32)

    def get_side_image(self, cam_name: str, front_frame_idx: int) -> np.ndarray | None:
        """Return side camera image nearest in time to front_frame_idx, or None if not present."""
        ts_side = self._side_ts.get(cam_name)
        if ts_side is None:
            idx = front_frame_idx
        else:
            idx = _nearest_idx(ts_side, self.get_timestamp(front_frame_idx))
        img_path = self.mission_dir / "images" / cam_name / f"{idx:06d}.jpeg"
        if not img_path.exists():
            return None
        return np.array(Image.open(img_path).convert("RGB"))
