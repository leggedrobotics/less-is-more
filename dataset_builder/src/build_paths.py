"""Build path zarr groups from Grand Tour missions.

Supports two dataset types:
  geo - MPPI-planned geometric paths (D_geo)
  tel - Imitation paths from the robot's recorded trajectory (D_tel)

Both types write the same zarr schema under data/{geometric,teleop}_paths/.

Usage
-----
  uv run dataset_builder/src/build_paths.py dataset_type=geo
  uv run dataset_builder/src/build_paths.py dataset_type=geo viz=true
  uv run dataset_builder/src/build_paths.py --config-name build_debug dataset_type=geo viz=true
"""

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from omegaconf import DictConfig
from tqdm import tqdm

from dataset_builder.helpers.transform_helpers import transform_se2_odom_to_base, convert_se2_to_transform
from dataset_builder.mppi_planner.mppi_planner import GridMap2D, MPPIPlanner
from dataset_builder.src.mission_data_source import GrandTourZarrSource
from utils.grandtour_hub import HF_REVISION_MAIN, HF_REVISION_LIMO, pull_mission_topics

log = logging.getLogger(__name__)

_GOAL_MEAN = np.array([5.0, 0.0, 0.0])
_GOAL_COV  = np.diag([2.5**2, 2.0**2, (np.pi / 4) ** 2])


def _resolve_paths(cfg: DictConfig) -> None:
    root = Path(__file__).resolve().parents[2]
    p = Path(cfg["dataset_folder"])
    if not p.is_absolute():
        cfg["dataset_folder"] = str(root / p)


def _write_zarr(zarr_dir: Path, paths, goals, image_ids, goal_times) -> None:
    zarr_dir.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(zarr_dir), mode="w")
    g.create_dataset("path",      data=np.array(paths,      dtype=np.float32), chunks=(1000, 50, 3))
    g.create_dataset("goal",      data=np.array(goals,      dtype=np.float32), chunks=(1000, 3))
    g.create_dataset("image_id",  data=np.array(image_ids,  dtype=np.int64),   chunks=(1000,))
    g.create_dataset("goal_time", data=np.array(goal_times, dtype=np.float32), chunks=(1000,))


# ── D_geo ──────────────────────────────────────────────────────────────────────

def _sample_geo_goal(rng: np.random.Generator) -> np.ndarray:
    pose = rng.multivariate_normal(_GOAL_MEAN, _GOAL_COV).astype(np.float32)
    pose[0] = abs(pose[0])
    return pose


def _check_min_nan_dist(elev: np.ndarray, min_cells: int) -> bool:
    t = torch.from_numpy(elev)
    nan_mask = torch.isnan(t)
    if not nan_mask.any():
        return True
    H, W = t.shape
    rows, cols = torch.where(nan_mask)
    dist = torch.sqrt(((rows - H // 2).float() ** 2 + (cols - W // 2).float() ** 2))
    return dist.min().item() >= min_cells


def _build_geo_mission(source, mission_dir, planner, cfg, rng, device, viz=None) -> int:
    origin = torch.tensor([-cfg.map_size, -cfg.map_size], dtype=torch.float32, device=device)
    start  = torch.zeros(3, dtype=torch.float32, device=device)
    paths_list, goals_list, image_ids_list, goal_times_list = [], [], [], []

    for i in tqdm(range(len(source)), desc=mission_dir.name, leave=False):
        elev_np = source.get_elevation(i)
        if np.isnan(elev_np).mean() > cfg.max_nan_frac:
            continue
        if not _check_min_nan_dist(elev_np, cfg.min_nan_dist_cells):
            continue

        gm = GridMap2D(
            elevation=torch.from_numpy(elev_np).to(device).T,
            resolution=cfg.map_resolution,
            origin_xy=origin,
        )

        frame_paths, frame_goals = [], []
        for _ in range(cfg.paths_per_image):
            goal   = _sample_geo_goal(rng)
            states = planner.plan(gm, start, torch.from_numpy(goal).to(device))
            frame_paths.append(states.cpu().numpy().astype(np.float32))
            frame_goals.append(goal)
            paths_list.append(frame_paths[-1])
            goals_list.append(goal)
            image_ids_list.append(i)
            goal_times_list.append(float(cfg.goal_time))

        if viz is not None and i % viz["every"] == 0:
            if not plt.fignum_exists(viz["fig"].number):
                break
            from dataset_builder.src.visualize import draw_frame
            draw_frame(viz["axes"], viz["fig"], source, planner,
                       frame_paths, frame_goals, i, elev_np,
                       viz["rob_w"], viz["rob_h"], viz["cams"])
            viz["fig"].canvas.draw()
            plt.pause(viz["delay"])

    n = len(paths_list)
    if n > 0:
        _write_zarr(mission_dir / "data" / "geometric_paths",
                    paths_list, goals_list, image_ids_list, goal_times_list)
    return n


# ── D_tel ──────────────────────────────────────────────────────────────────────

def _build_tel_mission(source, mission_dir, cfg, rng) -> int:
    paths_list, goals_list, image_ids_list, goal_times_list = [], [], [], []

    for i in tqdm(range(len(source)), desc=mission_dir.name, leave=False):
        goal_time = max(abs(float(rng.normal(cfg.goal_time_mean, cfg.goal_time_std))), 0.5)

        traj_world = source.get_trajectory_world(i, duration=goal_time, n=50)
        pose_world = source.get_pose_se2_world(i)
        path_base  = transform_se2_odom_to_base(traj_world, convert_se2_to_transform(pose_world))
        goal_base  = path_base[-1]

        if np.linalg.norm(path_base[0, :2] - path_base[-1, :2]) < 0.1:
            continue

        paths_list.append(path_base.astype(np.float32))
        goals_list.append(goal_base.astype(np.float32))
        image_ids_list.append(i)
        goal_times_list.append(float(goal_time))

    n = len(paths_list)
    if n > 0:
        _write_zarr(mission_dir / "data" / "teleop_paths",
                    paths_list, goals_list, image_ids_list, goal_times_list)
    return n


# ── Main ───────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="../configs", config_name="build")
def main(cfg: DictConfig) -> None:
    _resolve_paths(cfg)

    dataset_type = cfg.get("dataset_type", "geo")
    assert dataset_type in ("geo", "tel"), f"dataset_type must be 'geo' or 'tel', got {dataset_type!r}"

    torch.manual_seed(cfg.seed)
    rng    = np.random.default_rng(cfg.seed)
    device = cfg.device

    missions     = list(cfg.missions)
    grandtour_dir = Path(cfg.dataset_folder) / "grandtour"
    grandtour_dir.mkdir(parents=True, exist_ok=True)

    planner = MPPIPlanner(cfg.mppi, device) if dataset_type == "geo" else None

    fp     = cfg.mppi.footprint[0]
    rob_w  = (fp[1][1] - fp[0][1]) / cfg.map_resolution
    rob_h  = (fp[1][0] - fp[0][0]) / cfg.map_resolution

    use_viz = cfg.get("viz", False)
    viz_ctx = None
    if use_viz:
        from dataset_builder.src.visualize import make_figure
        plt.ion()
        fig, axes = make_figure()
        viz_ctx = {
            "fig":   fig,
            "axes":  axes,
            "every": int(cfg.get("viz_every", 50)),
            "delay": float(cfg.get("viz_delay", 1.0)),
            "rob_w": rob_w,
            "rob_h": rob_h,
            "cams":  None,  # loaded per mission below
        }

    log.info(f"Building D_{dataset_type} for {len(missions)} mission(s)")
    total = 0
    for mission_ts in missions:
        mission_dir = grandtour_dir / mission_ts
        log.info(f"[{mission_ts}] pulling topics …")

        pull_mission_topics(
            missions=[mission_ts],
            topics=["hdr_front", "dlio_map_odometry"],
            dataset_folder=grandtour_dir,
            revision=HF_REVISION_MAIN,
            skip_existing=True,
        )
        if dataset_type == "geo":
            pull_mission_topics(
                missions=[mission_ts],
                topics=["elevation_map"],
                dataset_folder=grandtour_dir,
                revision=cfg.elevation_revision,
                skip_existing=True,
            )
        elif dataset_type == "tel":
            pull_mission_topics(
                missions=[mission_ts],
                topics=["teleop_paths"],
                dataset_folder=grandtour_dir,
                revision=HF_REVISION_LIMO,
                skip_existing=True,
            )

        source = GrandTourZarrSource(mission_dir, cfg.map_size, cfg.map_resolution)
        log.info(f"[{mission_ts}] {len(source)} frames")

        if viz_ctx is not None:
            from dataset_builder.src.visualize import load_cameras
            viz_ctx["cams"] = load_cameras(mission_dir)

        if dataset_type == "geo":
            n = _build_geo_mission(source, mission_dir, planner, cfg, rng, device, viz_ctx)
        else:
            n = _build_tel_mission(source, mission_dir, cfg, rng)

        log.info(f"[{mission_ts}] wrote {n} samples")
        total += n

    if use_viz:
        plt.ioff()

    log.info(f"Done - {total} total samples across {len(missions)} missions")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
