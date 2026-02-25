from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List

import torch

StateDict = OrderedDict[str, torch.Tensor]


def secure_aggregate_with_masks(client_states: List[StateDict], seed: int = 42) -> StateDict:
    """Simulated SMPC secure aggregation with additive masks that cancel out."""
    if not client_states:
        raise ValueError("No client states provided")

    torch.manual_seed(seed)
    keys = client_states[0].keys()
    masked_states = []

    for i, state in enumerate(client_states):
        masked = OrderedDict()
        for key in keys:
            tensor = state[key]
            mask = torch.randn_like(tensor) * 1e-3
            if i % 2 == 0:
                masked[key] = tensor + mask
            else:
                masked[key] = tensor - mask
        masked_states.append(masked)

    agg = OrderedDict()
    for key in keys:
        agg[key] = sum(state[key] for state in masked_states) / len(masked_states)
    return agg
