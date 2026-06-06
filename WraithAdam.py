"""WraithAdam optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.optim.optimizer import Optimizer, ParamsT

if TYPE_CHECKING:
    from collections.abc import Callable


class WraithAdam(Optimizer):
    """WraithAdam optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-10,
    ) -> None:  # standart eps is 1e-8 we make for more stable training
        """Initialize WraithAdam optimizer with learning rate, betas, and eps."""
        super().__init__(params, defaults={"lr": lr, "betas": betas, "eps": eps})

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

                m.mul_(betas[0]).add_(
                    grad, alpha=1 - betas[0]
                )  # m = m * betas[0] + grad * (1 - betas[0])
                v.mul_(betas[1]).addcmul_(
                    grad, grad, value=1 - betas[0]
                )  # than v = v * betas[1] + grad * grad * (1 - betas[1]) we make v = v * betas[0] + grad * grad * (1 - betas[0])
                m_hat = m / (1 - betas[0] ** t)  # m_hat = m / (1 - betas[0] ** t)
                v_hat = v / (1 - betas[1] ** t)  # v_hat = v / (1 - betas[1] ** t)
                p.data.sub_(
                    lr * m_hat / (v_hat.sqrt() + eps)
                )  # w = w - lr * m_hat / (v_hat.sqrt() + eps)

        return loss
