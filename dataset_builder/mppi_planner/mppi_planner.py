from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil, floor
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from dataset_builder.mppi_planner.fast_geodis_wrapper import fast_gdf_wrapper
from dataset_builder.mppi_planner.mppi_optimizer import MPPIOptimizer
from dataset_builder.mppi_planner.traversability_filter import get_filter_torch

log = logging.getLogger(__name__)


@dataclass
class GridMap2D:
    """Robot-centric elevation grid. Indexed as elevation[x_idx, y_idx]."""

    elevation: torch.Tensor  # (H, W) float32
    resolution: float
    origin_xy: torch.Tensor  # (2,) world (x, y) of cell (0, 0)
    frame_id: str = "base"

    def to(self, device) -> "GridMap2D":
        return GridMap2D(
            elevation=self.elevation.to(device),
            resolution=self.resolution,
            origin_xy=self.origin_xy.to(device),
            frame_id=self.frame_id,
        )


def world_to_map_idx(xy: torch.Tensor, gm: GridMap2D) -> torch.Tensor:
    i = torch.floor((xy[..., 0] - gm.origin_xy[0]) / gm.resolution)
    j = torch.floor((xy[..., 1] - gm.origin_xy[1]) / gm.resolution)
    return torch.stack([i, j], dim=-1).long()


def valid_mask(ij: torch.Tensor, gm: GridMap2D) -> torch.Tensor:
    H, W = gm.elevation.shape
    return (ij[..., 0] >= 0) & (ij[..., 0] < H) & (ij[..., 1] >= 0) & (ij[..., 1] < W)


def max_pool(grid: torch.Tensor, n: int) -> torch.Tensor:
    if n < 1:
        return grid
    g = grid.float().unsqueeze(0).unsqueeze(0)
    k = 2 * n + 1
    return (
        F.max_pool2d(g, kernel_size=k, stride=1, padding=n)
        .squeeze(0)
        .squeeze(0)
        .to(grid.dtype)
    )


def build_robot_footprint(rectangles, grid_resolution, device):
    if len(rectangles) == 0:
        return torch.zeros((0, 2), device=device)
    rects = np.array(rectangles)
    rects_norm = rects / grid_resolution
    indices = []
    for nr in range(rects_norm.shape[0]):
        x_fwd = ceil(rects_norm[nr, 1, 0])
        x_bwd = floor(rects_norm[nr, 0, 0])
        y_lft = ceil(rects_norm[nr, 1, 1])
        y_rgt = floor(rects_norm[nr, 0, 1])
        for x in np.arange(x_bwd, x_fwd):
            for y in np.arange(y_rgt, y_lft):
                indices.append([x, y])
    unique = list({tuple(p) for p in indices})
    return torch.tensor(unique, dtype=torch.float32, device=device) * grid_resolution


def clip_on_ray(bounds_hw, goal_ij: torch.Tensor) -> torch.Tensor:
    H, W = int(bounds_hw[0]), int(bounds_hw[1])
    if (0 <= goal_ij[0] < H) and (0 <= goal_ij[1] < W):
        return goal_ij
    device, dtype = goal_ij.device, goal_ij.dtype
    centre = torch.tensor([(H - 1) / 2.0, (W - 1) / 2.0], dtype=dtype, device=device)
    d = goal_ij.float() - centre
    dx, dy = d.tolist()
    candidates = []
    if dx > 0:
        candidates.append((H - 1 - centre[0]) / dx)
    elif dx < 0:
        candidates.append((0 - centre[0]) / dx)
    if dy > 0:
        candidates.append((W - 1 - centre[1]) / dy)
    elif dy < 0:
        candidates.append((0 - centre[1]) / dy)
    t = min(c for c in candidates if c >= 0)
    return (
        (centre + d * t)
        .round()
        .clamp(
            min=torch.tensor([0, 0], device=device),
            max=torch.tensor([H - 1, W - 1], device=device),
        )
        .to(dtype)
    )


def smallest_angle(yaw1: torch.Tensor, yaw2) -> torch.Tensor:
    yaw2 = torch.as_tensor(yaw2, device=yaw1.device, dtype=yaw1.dtype)
    d = yaw1 - yaw2
    return (torch.remainder(d + torch.pi, 2 * torch.pi) - torch.pi).abs()


class MPPIObjective:
    def __init__(self, cfg, device: str):
        """cfg: OmegaConf DictConfig (the mppi: section of build.yaml)."""
        self.device = torch.device(device)
        self.cfg = cfg
        self._start: Optional[torch.Tensor] = None
        self._goal: Optional[torch.Tensor] = None
        self._gm: Optional[GridMap2D] = None
        self._trav: Optional[torch.Tensor] = None
        self._gdf: Optional[torch.Tensor] = None
        self._nn = get_filter_torch(device).to(self.device)

    def set_map(self, gm: GridMap2D) -> None:
        self._gm = gm
        footprint = OmegaConf.to_container(self.cfg.footprint, resolve=True)
        self._robot_footprint = build_robot_footprint(
            footprint, self._gm.resolution, self.device
        )
        self._trav = self._compute_traversability()

    def set_observation(
        self, gm: GridMap2D, start: torch.Tensor, goal: torch.Tensor
    ) -> None:
        self._start = start
        self._goal = goal
        if gm is not self._gm:
            self.set_map(gm)
        self._gdf = self._compute_gdf()

    @property
    def trav(self) -> Optional[torch.Tensor]:
        return self._trav

    @property
    def gdf(self) -> Optional[torch.Tensor]:
        return self._gdf

    @torch.no_grad()
    def evaluate(self, population: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.zero_small_actions(population.clone())
        states = self.rollout(u)
        values = -self.states_cost(states, u).mean(dim=1)
        return values, states

    @torch.no_grad()
    def zero_small_actions(self, actions: torch.Tensor) -> torch.Tensor:
        lin_mask = torch.norm(actions[..., :2], dim=-1) < self.cfg.vel_lin_min
        ang_mask = torch.abs(actions[..., 2]) < self.cfg.vel_ang_min
        actions[..., :2][lin_mask] = 0
        actions[..., 2][ang_mask] = 0
        return actions

    @torch.no_grad()
    def rollout(self, population: torch.Tensor) -> torch.Tensor:
        """Integrate (vx, vy, wz) controls -> SE(2) states (P, H, 3)."""
        P, H, _ = population.shape
        dt = self.cfg.dt
        x0, y0, yaw0 = self._start
        vxvy = population[:, :, :2] * dt
        wz = population[:, :, 2] * dt
        yaw_cum = torch.cumsum(wz, dim=1)
        # Use pre-step yaw for rotation: the heading the robot has when executing
        # velocity command at step t is yaw before wz[t] is applied.
        yaw_rot = yaw0 + torch.cat(
            [torch.zeros(P, 1, device=wz.device), yaw_cum[:, :-1]], dim=1
        )
        yaw = yaw0 + yaw_cum  # output waypoint yaw (heading after each step)
        cos_y, sin_y = torch.cos(yaw_rot), torch.sin(yaw_rot)
        R = torch.stack(
            [torch.stack([cos_y, -sin_y], dim=-1), torch.stack([sin_y, cos_y], dim=-1)],
            dim=-2,
        )
        pos = torch.cumsum(torch.matmul(R, vxvy.unsqueeze(-1)).squeeze(-1), dim=1)
        pos[..., 0] += x0
        pos[..., 1] += y0
        return torch.cat([pos, yaw.unsqueeze(-1)], dim=-1)

    @torch.no_grad()
    def get_position_distance_to_goal(self, states_xy: torch.Tensor) -> torch.Tensor:
        goal_xy = self._goal[:2].to(states_xy.device)
        l2 = torch.norm(states_xy - goal_xy, dim=-1)

        ij = world_to_map_idx(states_xy, self._gm)
        gdf2d = self._gdf.squeeze(0) if self._gdf.dim() == 3 else self._gdf
        finite = torch.isfinite(gdf2d)
        fill = (
            gdf2d[finite].mean()
            if finite.any()
            else torch.tensor(0.0, device=gdf2d.device)
        )
        gdf2d = torch.where(finite, gdf2d, fill)

        ij = ij.to(device=gdf2d.device, dtype=torch.long)
        Hg, Wg = gdf2d.shape
        v = ij[..., 0].clamp(0, Hg - 1)
        u = ij[..., 1].clamp(0, Wg - 1)
        gdf = (
            gdf2d.reshape(-1)
            .index_select(0, (v * Wg + u).reshape(-1))
            .reshape(states_xy.shape[:-1])
        )
        return torch.max(l2, gdf)

    @torch.no_grad()
    def states_cost(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        trav_cost, trav_unknown = self.get_trav_cost(states)
        trav_cost = (
            trav_cost * cfg.traversability_cost
            + trav_unknown * cfg.traversability_unknown_cost
        )

        control = (
            torch.abs(actions[..., 0]) * cfg.action_cost_trans_forward
            + torch.abs(actions[..., 1]) * cfg.action_cost_trans_side
            + torch.abs(actions[..., 2]) * cfg.action_cost_rotation
        )

        dist = self.get_position_distance_to_goal(states[..., :2])
        pos_cost = torch.log1p(cfg.distance_pre_log_factor * dist) * cfg.position_cost
        delta = (pos_cost.mean() - cfg.position_cost_mean_limit).clamp(min=0)
        pos_cost = pos_cost - delta

        heading_offset = smallest_angle(states[..., 2], self._goal[2])
        hw = (
            torch.maximum(cfg.max_distance_for_heading_cost - dist, torch.tensor(0.0))
            .max(dim=0)
            .values
        )
        heading_cost = heading_offset * hw * cfg.heading_cost

        cost = pos_cost + control + heading_cost
        at_goal = (heading_offset < cfg.heading_cost_at_goal_tolerance) & (
            dist < cfg.position_cost_at_goal_tolerance
        )
        cost[at_goal] = -cfg.at_goal_reward
        cost += trav_cost
        return cost

    @torch.no_grad()
    def _compute_traversability(self) -> torch.Tensor:
        cfg = self.cfg
        elev = self._gm.elevation.to(self.device)
        trav = 1.0 - self._nn(elev.unsqueeze(0))
        nan_mask = torch.isnan(trav)

        trav[nan_mask] = 0
        trav = max_pool(trav, cfg.fatal_cells_buffer)
        trav[nan_mask] = torch.nan

        b = cfg.border_cells
        for sl in [np.s_[:, -b:], np.s_[:, :b], np.s_[-b:, :], np.s_[:b, :]]:
            trav[sl] = torch.nan

        t = trav[~nan_mask]
        orig = t.clone()
        slope = cfg.risky_value / (cfg.risky_th - cfg.safe_th)
        t[orig < cfg.safe_th] = 0
        cautious = (orig >= cfg.safe_th) & (orig <= cfg.risky_th)
        t[cautious] = (orig[cautious] - cfg.safe_th) * slope
        t[(orig > cfg.risky_th) & (orig < cfg.fatal_th)] = cfg.risky_value
        t[orig >= cfg.fatal_th] = cfg.fatal_value
        trav[~nan_mask] = t
        return trav

    @torch.no_grad()
    def _compute_gdf(self) -> torch.Tensor:
        cfg = self.cfg
        elev = self._gm.elevation.to(self.device)
        gdf_mask = (self._trav >= cfg.fatal_value).float()
        gdf_mask[torch.isnan(self._trav)] = 0.0
        gdf_mask = (
            (max_pool(gdf_mask, cfg.gdf_obstacle_buffer) > 0).float().unsqueeze(0)
        )
        goal_ij = clip_on_ray(
            elev.shape, world_to_map_idx(self._goal[None, :2], self._gm)[0]
        )
        return (
            fast_gdf_wrapper(
                gdf_mask.float()[None],
                int(goal_ij[0]),
                int(goal_ij[1]),
            )[0]
            * self._gm.resolution
        )

    @torch.no_grad()
    def get_trav_cost(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        P, H, _ = states.shape
        cells_xy = self._robot_footprint
        E = cells_xy.shape[0]

        xy, yaw = states[..., :2], states[..., 2]
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        R = torch.stack(
            [torch.stack([cos_y, -sin_y], dim=-1), torch.stack([sin_y, cos_y], dim=-1)],
            dim=-2,
        )

        fp = cells_xy.view(1, 1, E, 2).expand(P, H, E, 2)
        wp = (
            torch.matmul(fp.view(P * H, E, 2), R.view(P * H, 2, 2).transpose(-1, -2))
            + xy.view(P * H, 1, 2)
        ).view(P, H, E, 2)

        ij = world_to_map_idx(wp, self._gm)
        m = valid_mask(ij.view(-1, 2), self._gm).view(P, H, E)

        trav_vals = torch.zeros((P, H, E), device=states.device)
        unknown_bin = torch.zeros((P, H, E), device=states.device)

        if m.any():
            vij = ij[m]
            vals = self._trav[vij[:, 0], vij[:, 1]]
            unknown_bin[m] = torch.isnan(vals).float()
            trav_vals[m] = torch.nan_to_num(vals, 0.0)

        denom = m.float().sum(dim=2).clamp_min(1.0)
        return (trav_vals * m).sum(dim=2) / denom, (unknown_bin * m).sum(dim=2) / denom


class MPPIPlanner:
    def __init__(self, cfg, device: str):
        """cfg: OmegaConf DictConfig (the mppi: section of build.yaml)."""
        self.device = device
        self.objective = MPPIObjective(cfg, device)
        self.optimizer = MPPIOptimizer(cfg, device)

    @torch.no_grad()
    def plan(
        self, gridmap: GridMap2D, start: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """Returns (H, 3) SE(2) waypoints in the same frame as start/goal."""
        self.objective.set_observation(gridmap, start, goal)
        self.optimizer.reset()
        mean_u, best_u = self.optimizer.optimize(self.objective)
        u = best_u if best_u is not None else mean_u
        u = self.objective.zero_small_actions(u)
        return self.objective.rollout(u.unsqueeze(0))[0]
