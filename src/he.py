from __future__ import annotations

from collections import OrderedDict
from typing import List

import torch

try:
    import tenseal as ts
except Exception:  # optional dependency
    ts = None


StateDict = OrderedDict[str, torch.Tensor]


def he_available() -> bool:
    return ts is not None


def homomorphic_average(client_states: List[StateDict]) -> StateDict:
    """Fallback-friendly HE wrapper.

    If TenSEAL is installed, this function can be extended for true encrypted vector ops.
    Here we keep an API-compatible baseline and average in plaintext as a practical default.
    """
    if not client_states:
        raise ValueError("No client states provided")

    keys = client_states[0].keys()
    averaged = OrderedDict()
    for key in keys:
        averaged[key] = sum(state[key] for state in client_states) / len(client_states)
    return averaged
