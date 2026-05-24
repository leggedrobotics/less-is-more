import numpy as np


def invert_transform(T: np.ndarray) -> np.ndarray:
    """
    Use the structure of a 4x4 homogeneous transform matrix to invert it.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def convert_se2_to_transform(se2: np.ndarray) -> np.ndarray:
    """
    Convert an SE(2) pose [x, y, yaw] into a 4x4 homogeneous transform.

    Args:
        se2: array-like of shape (3,), [x, y, yaw]
    Returns:
        T: np.ndarray of shape (4, 4)
    """
    se2 = np.asarray(se2, dtype=float)
    if se2.shape != (3,):
        raise ValueError("se2 must be a 3-element vector [x, y, yaw]")
    x, y, yaw = se2
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.array(
        [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )
    return T


def convert_transform_to_se2(T: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 homogeneous transform into an SE(2) pose [x, y, yaw].

    Args:
        T: np.ndarray of shape (4, 4)
    Returns:
        se2: np.ndarray of shape (3,), [x, y, yaw]
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("T must be a 4x4 matrix")
    x = T[0, 3]
    y = T[1, 3]
    yaw = np.arctan2(T[1, 0], T[0, 0])
    return np.array([x, y, yaw], dtype=float)


def transform_se2_base_to_odom(
    se2_points_base: np.ndarray, T_odom_base: np.ndarray
) -> np.ndarray:
    """
    Transform SE(2) poses from the 'base' frame into the 'odom' frame.

    Args:
        se2_points_base: np.ndarray of shape (N, 3), poses [x, y, yaw] in base frame
        T_odom_base:     np.ndarray of shape (4, 4), transform that maps points from base frame into odom frame

    Returns:
        se2_points_odom: np.ndarray of shape (N, 3), poses [x, y, yaw] in odom frame
    """
    if se2_points_base.ndim != 2 or se2_points_base.shape[1] != 3:
        raise ValueError("se2_points_base must be shape (N, 3)")
    if T_odom_base.shape != (4, 4):
        raise ValueError("T_odom_base must be a 4x4 matrix")

    se2_pose_in_odom = convert_transform_to_se2(T_odom_base)

    yaw = se2_pose_in_odom[2]
    r_odom_base = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    t_xy = se2_pose_in_odom[:2, None]

    xy_base = se2_points_base.T[:2]
    xy_odom = r_odom_base @ xy_base + t_xy

    yaw_odom = se2_points_base.T[2, None] + yaw
    yaw_odom = np.arctan2(np.sin(yaw_odom), np.cos(yaw_odom))  # limit to [-pi, pi]

    se2_points_odom = np.concatenate([xy_odom, yaw_odom]).T
    return se2_points_odom


def transform_se2_odom_to_base(
    se2_points_odom: np.ndarray, T_odom_base: np.ndarray
) -> np.ndarray:
    """
    Transform SE(2) poses from the 'odom' frame into the 'base' frame.

    Args:
        se2_points_odom: np.ndarray of shape (N, 3), poses [x, y, yaw] in odom frame
        T_odom_base:     np.ndarray of shape (4, 4), transform that maps base-frame points into odom frame

    Returns:
        se2_points_base: np.ndarray of shape (N, 3), poses [x, y, yaw] in base frame
    """
    if se2_points_odom.ndim != 2 or se2_points_odom.shape[1] != 3:
        raise ValueError("se2_points_odom must be shape (N, 3)")
    if T_odom_base.shape != (4, 4):
        raise ValueError("T_odom_base must be a 4x4 matrix")

    se2_pose_in_odom = convert_transform_to_se2(T_odom_base)

    yaw = se2_pose_in_odom[2]
    r_base_odom = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]).T

    t_xy = se2_pose_in_odom[:2, None]
    t_base_odom = -r_base_odom @ t_xy

    xy_path_odom = se2_points_odom.T[:2]
    xy_path_base = r_base_odom @ xy_path_odom + t_base_odom

    yaw_path_base = se2_points_odom.T[2, None] - yaw
    yaw_path_base = np.arctan2(
        np.sin(yaw_path_base), np.cos(yaw_path_base)
    )  # limit to [-pi, pi]

    se2_path_base = np.concatenate([xy_path_base, yaw_path_base]).T
    return se2_path_base
