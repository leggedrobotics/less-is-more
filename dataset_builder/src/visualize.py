"""Visualization helpers for the dataset builder pipeline.

Imported by build_paths.py when viz=true.  Not a standalone entry point.

Layout
------
  Top:    hdr_left | hdr_front | hdr_right  (paths projected with fisheye model)
  Bottom: elevation | traversability | GDF  (paths + goal diamonds overlaid)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

_COLORS = ["#dd72d8", "#a4d8df", "#FF8A50", "#E64A19", "#0B3D91"]
_FRONT_CAM_YAML = (
    Path(__file__).resolve().parents[2] / "limo/configs/model/camera_info.yaml"
)
_Z_GROUND = -0.57  # ground height in robot base frame [m]


# ── Camera loading ─────────────────────────────────────────────────────────────


def _quat_to_E(q_xyzw: np.ndarray, t: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )
    E = np.eye(4)
    E[:3, :3] = R.T
    E[:3, 3] = -R.T @ t
    return E


def _load_front_cam(img_w: int, img_h: int) -> dict | None:
    if not _FRONT_CAM_YAML.exists():
        return None
    with open(_FRONT_CAM_YAML) as f:
        info = yaml.safe_load(f)
    sx = img_w / info["width"]
    sy = img_h / info["height"]
    K = np.array(info["K"], dtype=np.float64).reshape(3, 3)
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return {
        "K": K,
        "D": np.array(info["D"], dtype=np.float64),
        "E": np.array(info["E"], dtype=np.float64).reshape(4, 4),
    }


def _load_side_cam(mission_dir: Path, cam: str, img_w: int, img_h: int) -> dict | None:
    ext_path = mission_dir / "metadata" / f"{cam}.yaml"
    int_path = mission_dir / "metadata" / f"{cam}_caminfo.yaml"
    if not ext_path.exists() or not int_path.exists():
        return None
    with open(ext_path) as f:
        ext = yaml.safe_load(f)
    with open(int_path) as f:
        ci = yaml.safe_load(f)["camera_info"]
    q = ext["transform"]["rotation"]
    t = ext["transform"]["translation"]
    E = _quat_to_E(
        np.array([q["x"], q["y"], q["z"], q["w"]]), np.array([t["x"], t["y"], t["z"]])
    )
    K = np.array(ci["K"], dtype=np.float64).reshape(3, 3)
    orig_w, orig_h = ci.get("width", img_w), ci.get("height", img_h)
    K[0, 0] *= img_w / orig_w
    K[0, 2] *= img_w / orig_w
    K[1, 1] *= img_h / orig_h
    K[1, 2] *= img_h / orig_h
    # Auto-detect wrong z direction: point 5 m left should be in front of hdr_left
    test_z = (E @ np.array([0, 5, 0, 1]))[2]
    if (test_z > 0) != (cam == "hdr_left"):
        E = np.diag([-1.0, -1.0, -1.0, 1.0]) @ E
    return {"K": K, "D": np.array(ci["D"], dtype=np.float64), "E": E}


def load_cameras(mission_dir: Path, img_w: int = 1920, img_h: int = 1280) -> dict:
    """Return {front, left, right} camera dicts (K, D, E).  Any missing -> None."""
    return {
        "front": _load_front_cam(img_w, img_h),
        "left": _load_side_cam(mission_dir, "hdr_left", img_w, img_h),
        "right": _load_side_cam(mission_dir, "hdr_right", img_w, img_h),
    }


# ── Fisheye projection ─────────────────────────────────────────────────────────


def _project_fisheye(xyz_base: np.ndarray, cam: dict, img_shape: tuple) -> np.ndarray:
    """Equidistant fisheye projection.  Returns (N, 2) pixels, NaN where not visible."""
    ones = np.ones((xyz_base.shape[0], 1))
    xyz_cam = (cam["E"] @ np.hstack([xyz_base, ones]).T).T[:, :3]
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    in_front = z > 0.1
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r, np.where(in_front, z, 1.0))
    k1, k2, k3, k4 = cam["D"][:4]
    th2 = theta**2
    rd = theta * (1 + k1 * th2 + k2 * th2**2 + k3 * th2**3 + k4 * th2**4)
    safe_r = np.where(r > 1e-9, r, 1.0)
    u = cam["K"][0, 0] * rd * (x / safe_r) + cam["K"][0, 2]
    v = cam["K"][1, 1] * rd * (y / safe_r) + cam["K"][1, 2]
    H, W = img_shape
    visible = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    px = np.full((len(xyz_base), 2), np.nan)
    px[visible, 0] = u[visible]
    px[visible, 1] = v[visible]
    return px


# ── Drawing helpers ────────────────────────────────────────────────────────────


def _path_to_px(
    path: np.ndarray, resolution: float, n_cells: int
) -> tuple[np.ndarray, np.ndarray]:
    c = n_cells // 2
    col = (c - 1) - path[:, 1] / resolution
    row = (c - 1) - path[:, 0] / resolution
    return col, row


def _draw_map(ax, data: np.ndarray, cmap: str, vmin=None, vmax=None, title="") -> None:
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="#cccccc")
    ax.imshow(
        np.rot90(data, 2),
        cmap=cm,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _overlay_maps(
    axes,
    paths,
    goals,
    colors,
    rob_w: float,
    rob_h: float,
    resolution: float,
    n_cells: int,
) -> None:
    c = n_cells // 2
    for ax in axes:
        for path, goal, col in zip(paths, goals, colors):
            px_col, px_row = _path_to_px(path, resolution, n_cells)
            ax.plot(px_col, px_row, "-", color=col, linewidth=3.0, alpha=0.9)
        for goal, col in zip(goals, colors):
            gcol, grow = _path_to_px(goal[None], resolution, n_cells)
            ax.plot(gcol[0], grow[0], "D", color=col, markersize=10, zorder=5)
        ax.add_patch(
            Rectangle(
                (c - 1 - rob_w / 2, c - 1 - rob_h / 2),
                rob_w,
                rob_h,
                linewidth=1,
                edgecolor="red",
                facecolor="red",
                alpha=0.85,
                zorder=6,
            )
        )


def _overlay_cam(ax, img_shape, paths, goals, colors, cam_info) -> None:
    if cam_info is None:
        return
    for path, goal, c in zip(paths, goals, colors):
        xyz = np.column_stack([path[:, 0], path[:, 1], np.full(len(path), _Z_GROUND)])
        px = _project_fisheye(xyz, cam_info, img_shape)
        ax.plot(px[:, 0], px[:, 1], "-", color=c, linewidth=3.0, alpha=0.9)
    for goal, c in zip(goals, colors):
        px_g = _project_fisheye(
            np.array([[goal[0], goal[1], _Z_GROUND]]), cam_info, img_shape
        )
        if not np.isnan(px_g[0, 0]):
            ax.plot(px_g[0, 0], px_g[0, 1], "D", color=c, markersize=10, zorder=5)


# ── Public API ─────────────────────────────────────────────────────────────────


def make_figure() -> tuple:
    """Create the persistent 2x3 figure.  Returns (fig, axes)."""
    fig = plt.figure(figsize=(20, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.08, wspace=0.04)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    return fig, axes


def draw_frame(
    axes,
    fig,
    source,
    planner,
    paths: list,  # list of (50, 3) float32 arrays
    goals: list,  # list of (3,) float32
    frame_idx: int,
    elev_np: np.ndarray,
    rob_w: float,  # robot width in pixels (= metres / resolution)
    rob_h: float,
    cams: dict,  # output of load_cameras()
    resolution: float = 0.04,
    n_cells: int = 200,
) -> None:
    """Clear axes and render one frame in-place."""
    ax_left, ax_front, ax_right, ax_el, ax_tr, ax_gd = axes
    for ax in axes:
        ax.cla()

    colors = [_COLORS[k % len(_COLORS)] for k in range(len(paths))]
    ts = source.get_timestamp(frame_idx)

    img_left = source.get_side_image("hdr_left", frame_idx)
    img_front = source.get_image(frame_idx)
    img_right = source.get_side_image("hdr_right", frame_idx)

    for ax, img, title, cam in [
        (ax_left, img_left, "hdr_left", cams["left"] if cams else None),
        (ax_front, img_front, "hdr_front", cams["front"] if cams else None),
        (ax_right, img_right, "hdr_right", cams["right"] if cams else None),
    ]:
        ax.axis("off")
        ax.set_title(title, fontsize=9)
        if img is None:
            continue
        ax.imshow(img)
        if cam is not None:
            _overlay_cam(ax, img.shape[:2], paths, goals, colors, cam)

    vmin = float(np.nanpercentile(elev_np, 2))
    vmax = float(np.nanpercentile(elev_np, 98))
    _draw_map(ax_el, elev_np, "terrain", vmin=vmin, vmax=vmax, title="Elevation")
    _overlay_maps([ax_el], paths, goals, colors, rob_w, rob_h, resolution, n_cells)

    if planner is not None:
        trav_np = planner.objective.trav.squeeze().cpu().numpy()
        gdf_np = planner.objective.gdf.squeeze().cpu().numpy()
        gdf_fin = gdf_np[np.isfinite(gdf_np)]
        gdf_max = float(np.percentile(gdf_fin, 95)) if len(gdf_fin) else 10.0
        _draw_map(ax_tr, trav_np, "coolwarm", vmin=0, vmax=1, title="Traversability")
        _draw_map(
            ax_gd,
            np.where(np.isfinite(gdf_np), gdf_np, np.nan),
            "viridis_r",
            vmin=0,
            vmax=gdf_max,
            title="GDF",
        )
        _overlay_maps(
            [ax_tr, ax_gd], paths, goals, colors, rob_w, rob_h, resolution, n_cells
        )
    else:
        ax_tr.axis("off")
        ax_gd.axis("off")

    fig.suptitle(
        f"{source.mission_dir.name}   frame {frame_idx}   t={ts:.3f} s", fontsize=9
    )
