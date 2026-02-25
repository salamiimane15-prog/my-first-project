from __future__ import annotations

import torch


def apply_dp_to_gradients(
    model: torch.nn.Module,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 0.8,
) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    for param in model.parameters():
        if param.grad is None:
            continue
        noise = torch.normal(
            mean=0.0,
            std=noise_multiplier * max_grad_norm,
            size=param.grad.shape,
            device=param.grad.device,
        )
        param.grad.add_(noise)
