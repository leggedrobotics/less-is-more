"""Build path zarr groups from Grand Tour missions.

Supports two dataset types:
  geo - MPPI-planned geometric paths (D_geo)
  tel - Imitation paths from the robot's recorded trajectory (D_tel)

Both types write the same zarr schema under data/{geometric,teleop}_paths/.

Usage
-----
  uv run dataset_builder/src/build_paths.py dataset_type=geo
  uv run dataset_builder/src/build_paths.py dataset_type=geo viz=true
  uv run dataset_builder/src/build_paths.py --config-name build_example dataset_type=geo viz=true
"""

import logging
import shutil
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import zarr
from omegaconf import DictConfig
from tqdm import tqdm

from dataset_builder.helpers.transform_helpers import (
    transform_se2_odom_to_base,
    convert_se2_to_transform,
)
from dataset_builder.mppi_planner.mppi_planner import GridMap2D, MPPIPlanner
from dataset_builder.src.mission_data_source import GrandTourZarrSource
from utils.grandtour_hub import HF_REVISION_MAIN, pull_mission_topics

log = logging.getLogger(__name__)


def _resolve_paths(cfg: DictConfig) -> None:
    root = Path(__file__).resolve().parents[2]
    p = Path(cfg.dataset_folder)
    if not p.is_absolute():
        cfg.dataset_folder = str(root / p)


def _write_zarr(
    zarr_dir: Path, paths, goals, image_ids, goal_times, write_mode: str = "overwrite"
) -> None:
    zarr_dir.mkdir(parents=True, exist_ok=True)
    if write_mode == "overwrite":
        g = zarr.open_group(str(zarr_dir), mode="w")
        g.create_dataset(
            "path", data=np.array(paths, dtype=np.float32), chunks=(1000, 50, 3)
        )
        g.create_dataset(
            "goal", data=np.array(goals, dtype=np.float32), chunks=(1000, 3)
        )
        g.create_dataset(
            "image_id", data=np.array(image_ids, dtype=np.int64), chunks=(1000,)
        )
        g.create_dataset(
            "goal_time", data=np.array(goal_times, dtype=np.float32), chunks=(1000,)
        )
    else:  # append
        g = zarr.open_group(str(zarr_dir), mode="a")
        new_paths = np.array(paths, dtype=np.float32)
        new_goals = np.array(goals, dtype=np.float32)
        new_image_ids = np.array(image_ids, dtype=np.int64)
        new_goal_times = np.array(goal_times, dtype=np.float32)
        if "path" in g:
            g["path"].append(new_paths)
            g["goal"].append(new_goals)
            g["image_id"].append(new_image_ids)
            g["goal_time"].append(new_goal_times)
        else:
            g.create_dataset("path", data=new_paths, chunks=(1000, 50, 3))
            g.create_dataset("goal", data=new_goals, chunks=(1000, 3))
            g.create_dataset("image_id", data=new_image_ids, chunks=(1000,))
            g.create_dataset("goal_time", data=new_goal_times, chunks=(1000,))


# ── D_geo ──────────────────────────────────────────────────────────────────────


def _sample_geo_goal(rng: np.random.Generator, cfg) -> np.ndarray:
    mean = np.array([cfg.goal_x_mean, cfg.goal_y_mean, cfg.goal_yaw_mean])
    cov = np.diag([cfg.goal_x_std**2, cfg.goal_y_std**2, cfg.goal_yaw_std**2])
    pose = rng.multivariate_normal(mean, cov).astype(np.float32)
    return pose


def _check_min_nan_dist(elev: np.ndarray, min_cells: int) -> bool:
    nan_mask = np.isnan(elev)
    if not nan_mask.any():
        return True
    H, W = elev.shape
    rows, cols = np.where(nan_mask)
    dist = np.sqrt(((rows - H // 2) ** 2 + (cols - W // 2) ** 2).astype(float))
    return float(dist.min()) >= min_cells


def _build_mission(
    source, mission_dir, zarr_subdir, cfg, write_mode, viz, step_fn
) -> tuple[int, bool]:
    """Run the per-frame loop and write zarr output.

    step_fn(i) -> (records, viz_data)
      records:  list of (path, goal, goal_time) tuples — empty = skip frame
      viz_data: (elev_np, planner_or_none) used when viz triggers, or None
    """
    paths_list, goals_list, image_ids_list, goal_times_list = [], [], [], []
    skip_first = int(cfg.get("skip_first_frames", 0))
    interrupted = False
    try:
        for i in tqdm(
            range(skip_first, len(source)), desc=mission_dir.name, leave=False
        ):
            if viz is not None:
                if not viz["plt"].fignum_exists(viz["fig"].number):
                    break
                viz["fig"].canvas.flush_events()

            records, viz_data = step_fn(i)
            for path, goal, goal_time in records:
                paths_list.append(path)
                goals_list.append(goal)
                image_ids_list.append(i)
                goal_times_list.append(goal_time)

            if viz is not None and records and i % viz["every"] == 0:
                from dataset_builder.src.visualize import draw_frame

                elev_np, planner_for_viz = viz_data
                draw_frame(
                    viz["axes"],
                    viz["fig"],
                    source,
                    planner_for_viz,
                    [r[0] for r in records],
                    [r[1] for r in records],
                    i,
                    elev_np,
                    viz["rob_w"],
                    viz["rob_h"],
                    viz["cams"],
                    viz["resolution"],
                    viz["n_cells"],
                )
                viz["fig"].canvas.draw()
                viz["plt"].pause(viz["delay"])
    except KeyboardInterrupt:
        interrupted = True
    finally:
        n = len(paths_list)
        if n > 0:
            _write_zarr(
                mission_dir / "data" / zarr_subdir,
                paths_list,
                goals_list,
                image_ids_list,
                goal_times_list,
                write_mode,
            )
    return len(paths_list), interrupted


def _build_geo_mission(
    source, mission_dir, planner, cfg, rng, device, write_mode, viz=None
) -> tuple[int, bool]:
    origin = torch.tensor(
        [-cfg.map_size, -cfg.map_size], dtype=torch.float32, device=device
    )
    start = torch.zeros(3, dtype=torch.float32, device=device)

    def step_fn(i):
        elev_np = source.get_elevation(i)
        if np.isnan(elev_np).mean() > cfg.max_nan_frac:
            return [], None
        if not _check_min_nan_dist(elev_np, cfg.min_nan_dist_cells):
            return [], None
        gm = GridMap2D(
            elevation=torch.from_numpy(elev_np).to(device),
            resolution=cfg.map_resolution,
            origin_xy=origin,
        )
        records = []
        for _ in range(cfg.paths_per_image):
            goal = _sample_geo_goal(rng, cfg)
            states = planner.plan(gm, start, torch.from_numpy(goal).to(device))
            records.append(
                (states.cpu().numpy().astype(np.float32), goal, float(cfg.goal_time))
            )
        return records, (elev_np, planner)

    return _build_mission(
        source, mission_dir, "geometric_paths", cfg, write_mode, viz, step_fn
    )


# ── D_tel ──────────────────────────────────────────────────────────────────────


def _build_tel_mission(
    source, mission_dir, cfg, rng, write_mode, viz=None
) -> tuple[int, bool]:
    def step_fn(i):
        goal_time = max(
            abs(float(rng.normal(cfg.goal_time_mean, cfg.goal_time_std))), 0.5
        )
        traj_world = source.get_trajectory_world(i, duration=goal_time, n=50)
        pose_world = source.get_pose_se2_world(i)
        path_base = transform_se2_odom_to_base(
            traj_world, convert_se2_to_transform(pose_world)
        )
        goal_base = path_base[-1]
        if np.linalg.norm(path_base[0, :2] - path_base[-1, :2]) < 0.1:
            return [], None
        elev_np = source.get_elevation(i) if viz is not None else None
        return [
            (
                path_base.astype(np.float32),
                goal_base.astype(np.float32),
                float(goal_time),
            )
        ], (elev_np, None)

    return _build_mission(
        source, mission_dir, "teleop_paths", cfg, write_mode, viz, step_fn
    )


# ── Main ───────────────────────────────────────────────────────────────────────


@hydra.main(version_base="1.3", config_path="../configs", config_name="build")
def main(cfg: DictConfig) -> None:
    _resolve_paths(cfg)

    dataset_type = cfg.get("dataset_type", "geo")
    if dataset_type not in ("geo", "tel"):
        raise ValueError(f"dataset_type must be 'geo' or 'tel', got {dataset_type!r}")

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = cfg.device

    missions = list(cfg.missions)
    grandtour_dir = Path(cfg.dataset_folder) / "grandtour"
    grandtour_dir.mkdir(parents=True, exist_ok=True)

    write_mode = cfg.get("write_mode", "overwrite")
    zarr_subdir = "geometric_paths" if dataset_type == "geo" else "teleop_paths"

    if write_mode == "overwrite":
        existing = [
            m for m in missions if (grandtour_dir / m / "data" / zarr_subdir).exists()
        ]
        if existing:
            log.warning(
                f"{len(existing)} mission(s) already have '{zarr_subdir}' data:"
            )
            for m in existing:
                log.warning(f"  {m}")
            if sys.stdin.isatty():
                answer = (
                    input("Delete existing data and rebuild from scratch? [y/N]: ")
                    .strip()
                    .lower()
                )
            else:
                log.warning(
                    "Non-interactive mode: aborting. Use write_mode=append or remove existing data manually."
                )
                return
            if answer != "y":
                log.info("Aborted. Use write_mode=append to add to existing data.")
                return
            for m in existing:
                shutil.rmtree(grandtour_dir / m / "data" / zarr_subdir)
                log.info(f"Removed {m}/data/{zarr_subdir}")

    planner = MPPIPlanner(cfg.mppi, device) if dataset_type == "geo" else None

    use_viz = cfg.get("viz", False)
    viz_ctx = None

    # Download all missions before opening the figure so there is no blank
    # unresponsive window during network I/O.
    topics_main = ["hdr_front", "dlio_map_odometry"]
    if cfg.get("fetch_side_cams"):
        topics_main += ["hdr_left", "hdr_right"]

    log.info(f"Pulling data for {len(missions)} mission(s) ...")
    pull_mission_topics(
        missions=missions,
        topics=topics_main,
        dataset_folder=grandtour_dir,
        revision=HF_REVISION_MAIN,
        skip_existing=True,
    )
    if dataset_type == "geo":
        pull_mission_topics(
            missions=missions,
            topics=["elevation_map"],
            dataset_folder=grandtour_dir,
            revision=cfg.elevation_revision,
            skip_existing=True,
        )

    if use_viz:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from dataset_builder.src.visualize import make_figure

        fp = cfg.mppi.footprint[0]
        rob_w = (fp[1][1] - fp[0][1]) / cfg.map_resolution
        rob_h = (fp[1][0] - fp[0][0]) / cfg.map_resolution
        plt.ion()
        fig, axes = make_figure()
        viz_ctx = {
            "fig": fig,
            "axes": axes,
            "every": int(cfg.viz_every),
            "delay": float(cfg.get("viz_delay", 1.0)),
            "rob_w": rob_w,
            "rob_h": rob_h,
            "cams": None,
            "resolution": cfg.map_resolution,
            "n_cells": int(cfg.map_size * 2 / cfg.map_resolution),
            "plt": plt,
        }

    log.info(f"Building D_{dataset_type} for {len(missions)} mission(s)")
    total = 0
    for mission_ts in missions:
        mission_dir = grandtour_dir / mission_ts

        source = GrandTourZarrSource(mission_dir, cfg.map_size, cfg.map_resolution)
        log.info(f"[{mission_ts}] {len(source)} frames")

        if viz_ctx is not None:
            from dataset_builder.src.visualize import load_cameras

            viz_ctx["cams"] = load_cameras(mission_dir)

        if dataset_type == "geo":
            n, interrupted = _build_geo_mission(
                source, mission_dir, planner, cfg, rng, device, write_mode, viz_ctx
            )
        else:
            n, interrupted = _build_tel_mission(
                source, mission_dir, cfg, rng, write_mode, viz_ctx
            )

        log.info(f"[{mission_ts}] wrote {n} samples")
        total += n
        if interrupted:
            log.info("Interrupted.")
            break

    if use_viz:
        plt.ioff()

    log.info(f"Done - {total} total samples across {len(missions)} missions")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
