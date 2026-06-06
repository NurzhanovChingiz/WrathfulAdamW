"""WraithAdamW optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.optim.optimizer import Optimizer, ParamsT

if TYPE_CHECKING:
    from collections.abc import Callable


class WraithAdamW(Optimizer):
    """WraithAdamW optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-10,
        weight_decay: float = 0.01,
    ) -> None:  # standart eps is 1e-8 we make for more stable training
        """Initialize WraithAdamW optimizer with learning rate, betas, and eps."""
        super().__init__(
            params,
            defaults={
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        """Perform a single optimization step.

        Args:
            closure: A closure that evaluates the loss.

        Returns:
            The loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.clone()
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]
                p.data.mul_(1 - lr * weight_decay)  # w = w * (1 - lr * weight_decay)
                m.mul_(betas[0]).add_(
                    grad, alpha=1 - betas[0]
                )  # m = beta1 * m + (1 - beta1) * grad
                v.mul_(betas[1]).addcmul_(
                    grad, grad, value=1 - betas[0]
                )  # than v = beta2 * v + (1 - beta2) * grad**2 we make v = beta2 * v + (1 - beta1) * grad**2

                m_hat = m / (1 - betas[0] ** t)  # m_hat = m / (1 - betas[0] ** t)
                v_hat = v / (1 - betas[1] ** t)  # v_hat = v / (1 - betas[1] ** t)
                p.data.sub_(
                    lr * m_hat / (v_hat.sqrt() + eps)
                )  # w = w - lr * m_hat / (v_hat.sqrt() + eps)

        return loss
