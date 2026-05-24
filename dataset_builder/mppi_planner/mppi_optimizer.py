from typing import Optional, Tuple

import torch


def truncated_normal_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, max_iters: int = 100
) -> torch.Tensor:
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    for _ in range(max_iters):
        cond = (tensor < mean - 2 * std) | (tensor > mean + 2 * std)
        if not cond.any():
            break
        tensor[cond] = torch.normal(
            mean, std, size=(cond.sum().item(),), device=tensor.device
        )
    return tensor


class MPPIOptimizer:
    def __init__(self, cfg, device: str):
        """cfg: OmegaConf DictConfig (the mppi: section of build.yaml)."""
        self.cfg = cfg
        self.device = torch.device(device)
        self._allocate_from_cfg()

    def reset(self) -> None:
        self.mean.zero_()

    @torch.no_grad()
    def optimize(
        self,
        objective,
        previous_population: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if previous_population is not None:
            if previous_population.dim() == 3:
                self.mean = previous_population[0].to(self.device)
            elif previous_population.dim() == 2:
                self.mean = previous_population.to(self.device)

        past_action = self.mean[0].clone()
        if self.cfg.horizon > 1:
            self.mean[:-1] = self.mean[1:].clone()

        best_traj = None
        best_value = -torch.inf

        H, A = self.cfg.horizon, 3
        P = self.cfg.population_size - (1 if self.cfg.provide_zero_action else 0)

        for _ in range(self.cfg.num_iterations):
            noise = truncated_normal_(torch.empty((P, H, A), device=self.device))

            lb_dist = self.mean - self.lower_bound
            ub_dist = self.upper_bound - self.mean
            mv = torch.minimum((lb_dist / 2) ** 2, (ub_dist / 2) ** 2)
            constrained_var = torch.minimum(mv, self.var)

            population = noise * torch.sqrt(constrained_var).unsqueeze(0)
            population[:, 0, :] += (
                self.cfg.beta * self.mean[0, :] + (1.0 - self.cfg.beta) * past_action
            )
            for t in range(H - 1):
                population[:, t + 1, :] += (
                    self.cfg.beta * self.mean[t + 1, :]
                    + (1.0 - self.cfg.beta) * population[:, t, :]
                )

            population = torch.clamp(
                population, self.lower_bound.unsqueeze(0), self.upper_bound.unsqueeze(0)
            )

            if self.cfg.provide_zero_action:
                population = torch.cat(
                    [population, torch.zeros((1, H, A), device=self.device)], dim=0
                )

            values, _ = objective.evaluate(population)
            values[torch.isnan(values)] = -1e10

            v_max, idx = torch.max(values, dim=0)
            if v_max > best_value:
                best_value = v_max
                best_traj = population[idx].clone()

            weights = torch.exp(self.cfg.gamma * (values - values.max()))
            weights = weights.view(-1, 1, 1)
            self.mean = (population * weights).sum(dim=0) / (weights.sum() + 1e-10)

        return self.mean.clone(), best_traj

    def _allocate_from_cfg(self) -> None:
        c = self.cfg
        H, D = c.horizon, 3

        self.mean = torch.zeros((H, D), device=self.device, dtype=torch.float32)

        lb_raw = (
            list(c.lower_bound) if hasattr(c.lower_bound, "__iter__") else c.lower_bound
        )
        ub_raw = (
            list(c.upper_bound) if hasattr(c.upper_bound, "__iter__") else c.upper_bound
        )
        lb = torch.as_tensor(lb_raw, device=self.device, dtype=torch.float32)
        ub = torch.as_tensor(ub_raw, device=self.device, dtype=torch.float32)
        if lb.ndim == 0:
            lb = lb.repeat(D)
        if ub.ndim == 0:
            ub = ub.repeat(D)
        self.lower_bound = lb.unsqueeze(0).repeat(H, 1)
        self.upper_bound = ub.unsqueeze(0).repeat(H, 1)
        self.var = (c.sigma**2) * torch.ones_like(self.mean)
