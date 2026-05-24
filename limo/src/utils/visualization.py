"""Visualization utilities for path planning and prediction."""

from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def project_path_to_image(
    path: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
    image_shape: Tuple[int, int],
    z_robot: float = -0.57,
) -> np.ndarray:
    """
    Project SE2 path points (x, y, yaw) to image pixels.

    Args:
        path: (N, 3) array of (x, y, yaw) in robot base frame
        K: (3, 3) camera intrinsic matrix
        D: (4,) or (5,) distortion coefficients
        E: (4, 4) camera extrinsic matrix (T_camera_base)
        image_shape: (height, width)
        z_robot: height of robot base frame above ground (default -0.57 for 57cm below base)

    Returns:
        pixels: (M, 2) array of pixel coordinates (M <= N, filtered for valid points)
    """
    xyz_robot = np.zeros((path.shape[0], 4))
    xyz_robot[:, 0] = path[:, 0]
    xyz_robot[:, 1] = path[:, 1]
    xyz_robot[:, 2] = z_robot
    xyz_robot[:, 3] = 1.0

    xyz_cam_homog = (E @ xyz_robot.T).T
    xyz_cam = xyz_cam_homog[:, :3]

    valid_mask = xyz_cam[:, 2] > 0.1
    xyz_cam_valid = xyz_cam[valid_mask]

    if len(xyz_cam_valid) == 0:
        return np.zeros((0, 2))

    h, w = image_shape
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0
    )

    rvec = np.zeros(3)
    tvec = np.zeros(3)
    pixels, _ = cv2.projectPoints(xyz_cam_valid, rvec, tvec, new_K, np.zeros_like(D))
    pixels = pixels.reshape(-1, 2)

    return pixels


def undistort_image(image: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Undistort image using fisheye model."""
    h, w = image.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1
    )
    undistorted = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return undistorted


def draw_diamond_bottom_anchor(
    ax: plt.Axes,
    x: float,
    y: float,
    size_pt: float,
    facecolor: str = "#a7cc6eff",
    edgecolor: str = "none",
    linewidth: float = 0.0,
    zorder: int = 6,
    width_scale: float = 0.65,
    alpha: float = 1.0,
) -> None:
    """Draw a diamond marker anchored at the bottom."""
    from matplotlib import colors as mcolors
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea
    from matplotlib.patches import Polygon

    h = float(size_pt)
    w = float(size_pt) * float(width_scale)
    da = DrawingArea(w, h, 0, 0)
    verts = [(w / 2.0, 0.0), (w, h / 2.0), (w / 2.0, h), (0.0, h / 2.0)]
    fc_rgba = (
        facecolor
        if isinstance(facecolor, (tuple, list))
        else mcolors.to_rgba(facecolor, alpha=alpha)
    )

    poly = Polygon(
        verts, closed=True, facecolor=fc_rgba, edgecolor=edgecolor, linewidth=linewidth
    )
    da.add_artist(poly)

    ab = AnnotationBbox(
        da,
        (x, y),
        xycoords="data",
        box_alignment=(0.5, 0.0),
        frameon=False,
        zorder=zorder,
        clip_on=True,
    )
    ax.add_artist(ab)


def create_bev_visualization(paths: List[np.ndarray], goals: np.ndarray) -> np.ndarray:
    """
    Create bird's eye view visualization.
    Robot is at bottom center, paths go upward toward goals.
    Goals outside the view are shown smaller on the borders.

    Args:
        paths: List of (N, 3) arrays of predicted paths (x, y, yaw)
        goals: (M, 3) array of goals (x, y, yaw)

    Returns:
        bev_image: RGB image as numpy array
    """
    view_radius = 8.0
    xlim = (-4.0, 4.0)
    ylim = (0, view_radius * 1.2)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("white")

    robot_y, robot_x = 0.0, 0.0

    from matplotlib.patches import Rectangle

    robot_width = 0.5
    robot_length = 0.8
    robot_rect = Rectangle(
        (robot_x - robot_width / 2, robot_y - robot_length / 2),
        robot_width,
        robot_length,
        facecolor="red",
        edgecolor="red",
        linewidth=2,
        zorder=1000,
    )
    ax.add_patch(robot_rect)

    path_colors = ["#dd72d8ff", "#a4d8dfff", "#FF8A50", "#0B3D91", "#E64A19"]

    for i, path in enumerate(paths):
        if len(path) > 0:
            color = path_colors[i % len(path_colors)]
            ys = path[:, 0]
            xs = -path[:, 1]
            ax.plot(xs, ys, "-", color=color, linewidth=8, alpha=1.0, zorder=10 + i)

    goal_size = 50.0
    border_goal_size = 20.0

    for i, goal in enumerate(goals):
        gx_fwd, gy_lat, gyaw = goal
        color = path_colors[i % len(path_colors)]
        goal_dist = np.hypot(gx_fwd, gy_lat)

        # Goal position in plot coordinates
        goal_x = -gy_lat
        goal_y = gx_fwd

        # Check if goal is within view bounds
        in_view_x = xlim[0] <= goal_x <= xlim[1]
        in_view_y = ylim[0] <= goal_y <= ylim[1]

        if in_view_x and in_view_y:
            # Draw goal with full size
            draw_diamond_bottom_anchor(
                ax,
                x=goal_x,
                y=goal_y,
                size_pt=goal_size,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                zorder=200.0 - goal_dist,
                width_scale=0.65,
                alpha=1.0,
            )
        else:
            # Clamp goal position to border
            clamped_x = np.clip(goal_x, xlim[0] + 0.3, xlim[1] - 0.3)
            clamped_y = np.clip(goal_y, ylim[0] + 0.3, ylim[1] - 0.3)

            # Draw smaller goal on border
            draw_diamond_bottom_anchor(
                ax,
                x=clamped_x,
                y=clamped_y,
                size_pt=border_goal_size,
                facecolor=color,
                edgecolor="white",
                linewidth=1.5,
                zorder=300,
                width_scale=0.65,
                alpha=0.8,
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    x_ticks = np.arange(-4.0, 4.0 + 0.1, 2.0)
    y_ticks = np.arange(0, view_radius * 1.2 + 0.1, 2.0)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{t:.0f} m" for t in x_ticks], fontsize=10)
    ax.set_yticklabels([f"{t:.0f} m" for t in y_ticks], fontsize=10)

    ax.tick_params(direction="out", width=1.0, color="#444", length=6)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.0)

    fig.tight_layout(pad=0)

    fig.canvas.draw()
    bev_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    bev_image = bev_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    bev_image = bev_image[:, :, :3]
    plt.close(fig)

    return bev_image


def create_camera_visualization(
    original_image: np.ndarray,
    paths: List[np.ndarray],
    goals: np.ndarray,
    camera_info: Optional[Dict],
) -> np.ndarray:
    """
    Create camera view visualization with projected paths.

    Args:
        original_image: (H, W, 3) RGB image
        paths: List of (N, 3) arrays of paths to project onto image
        goals: (M, 3) array of goals to project onto image
        camera_info: Dict with 'K', 'D', 'E' matrices, or None for raw image

    Returns:
        camera_panel: RGB image as numpy array
    """
    if camera_info is not None:
        K = np.array(camera_info["K"]).reshape(3, 3)
        D = np.array(camera_info["D"])
        E = np.array(camera_info["E"]).reshape(4, 4)

        img_display = undistort_image(original_image, K, D)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_display)
        ax.set_autoscale_on(False)
        h, w = img_display.shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        path_colors = ["#dd72d8ff", "#a4d8dfff", "#FF8A50", "#0B3D91", "#E64A19"]

        for i, path in enumerate(paths):
            color = path_colors[i % len(path_colors)]
            pixels = project_path_to_image(path, K, D, E, img_display.shape[:2])
            if len(pixels) > 1:
                ax.plot(
                    pixels[:, 0],
                    pixels[:, 1],
                    "-",
                    color=color,
                    linewidth=8,
                    alpha=1.0,
                    zorder=10 + i,
                )

        goal_size = 240.0

        for i, goal in enumerate(goals):
            goal_pos = np.array([[goal[0], goal[1], goal[2]]])
            goal_pixels = project_path_to_image(
                goal_pos, K, D, E, img_display.shape[:2]
            )
            if len(goal_pixels) > 0:
                color = path_colors[i % len(path_colors)]
                dist = np.hypot(goal[0], goal[1])
                size_pt = goal_size / max(dist, 0.15)
                draw_diamond_bottom_anchor(
                    ax,
                    x=float(goal_pixels[0, 0]),
                    y=float(goal_pixels[0, 1]),
                    size_pt=size_pt,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=200.0 - dist,
                    width_scale=0.65,
                    alpha=1.0,
                )

        ax.axis("off")

        fig.tight_layout(pad=0)
        fig.canvas.draw()
        camera_panel = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        camera_panel = camera_panel.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        camera_panel = camera_panel[:, :, :3]
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(original_image)
        ax.axis("off")
        ax.set_title("Input Image")
        fig.tight_layout(pad=0)

        fig.canvas.draw()
        camera_panel = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        camera_panel = camera_panel.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        camera_panel = camera_panel[:, :, :3]
        plt.close(fig)

    return camera_panel


def create_combined_visualization(
    original_image: np.ndarray,
    paths: List[np.ndarray],
    goals: np.ndarray,
    camera_info: Optional[Dict] = None,
    image_left: Optional[np.ndarray] = None,
    image_right: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create combined visualization with camera view(s) and BEV.

    Layout without side cams: [front | BEV]
    Layout with side cams:    [left | front | right | BEV]

    Args:
        original_image: (H, W, 3) RGB front image
        paths: List of (N, 3) arrays of predicted paths (x, y, yaw)
        goals: (M, 3) array of goals (x, y, yaw)
        camera_info: Dict with 'K', 'D', 'E' matrices, or None for raw image
        image_left: optional (H, W, 3) RGB left image
        image_right: optional (H, W, 3) RGB right image

    Returns:
        combined_image: RGB composite
    """
    front_panel = create_camera_visualization(original_image, paths, goals, camera_info)
    bev_panel = create_bev_visualization(paths, goals)

    h = front_panel.shape[0]
    pad = np.ones((h, 15, 3), dtype=np.uint8) * 255

    bev_w = int(h * bev_panel.shape[1] / bev_panel.shape[0])
    bev_resized = cv2.resize(bev_panel, (bev_w, h))

    panels = []
    if image_left is not None:
        side_w = int(h * image_left.shape[1] / image_left.shape[0])
        panels += [cv2.resize(image_left, (side_w, h)), pad]
    panels += [front_panel, pad]
    if image_right is not None:
        side_w = int(h * image_right.shape[1] / image_right.shape[0])
        panels += [cv2.resize(image_right, (side_w, h)), pad]
    panels += [bev_resized, pad]

    return np.hstack(panels)


def load_image_and_normalize(
    image_path: str, target_size: Tuple[int, int] = (308, 476)
) -> torch.Tensor:
    """
    Load image and normalize to tensor.

    Args:
        image_path: Path to image file
        target_size: (height, width) for resize

    Returns:
        image_tensor: (1, 3, H, W) normalized image tensor
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
    )
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor
