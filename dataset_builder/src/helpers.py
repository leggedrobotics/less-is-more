"""SE(2) transform utilities."""
import numpy as np


def convert_transform_to_se2(T: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform -> SE(2) [x, y, yaw]."""
    return np.array([T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])], dtype=float)


def convert_se2_to_transform(se2: np.ndarray) -> np.ndarray:
    """SE(2) [x, y, yaw] -> 4x4 homogeneous transform."""
    x, y, yaw = np.asarray(se2, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0, x],
        [s,  c, 0.0, y],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ], dtype=float)


def transform_se2_base_to_odom(se2_base: np.ndarray, T_odom_base: np.ndarray) -> np.ndarray:
    """Transform SE(2) poses (N, 3) from base frame to odom frame."""
    pose_odom = convert_transform_to_se2(T_odom_base)
    yaw = pose_odom[2]
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    t = pose_odom[:2, None]
    xy_odom = R @ se2_base.T[:2] + t
    yaw_odom = np.arctan2(np.sin(se2_base.T[2] + yaw), np.cos(se2_base.T[2] + yaw))
    return np.concatenate([xy_odom, yaw_odom[None]]).T


def transform_se2_odom_to_base(se2_odom: np.ndarray, T_odom_base: np.ndarray) -> np.ndarray:
    """Transform SE(2) poses (N, 3) from odom frame to base frame."""
    pose_odom = convert_transform_to_se2(T_odom_base)
    yaw = pose_odom[2]
    R_inv = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]).T
    t = -R_inv @ pose_odom[:2, None]
    xy_base = R_inv @ se2_odom.T[:2] + t
    yaw_base = np.arctan2(np.sin(se2_odom.T[2] - yaw), np.cos(se2_odom.T[2] - yaw))
    return np.concatenate([xy_base, yaw_base[None]]).T
