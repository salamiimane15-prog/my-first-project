from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from torch import optim

from src.data import DatasetBundle, partition_iid, partition_non_iid
from src.he import homomorphic_average
from src.models import build_model
from src.smpc import secure_aggregate_with_masks
from src.train_utils import evaluate, make_loader, train_one_epoch


def _average_state_dicts(state_dicts: List[OrderedDict[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
    keys = state_dicts[0].keys()
    avg = OrderedDict()
    for key in keys:
        avg[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    return avg


def run_federated_training(
    data: DatasetBundle,
    model_name: str = "resmlp",
    rounds: int = 15,
    num_clients: int = 10,
    local_epochs: int = 2,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    iid: bool = True,
    dirichlet_alpha: float = 0.3,
    use_smpc: bool = False,
    use_dp: bool = False,
    use_he: bool = False,
    dp_noise_multiplier: float = 0.8,
    dp_max_grad_norm: float = 1.0,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    global_model = build_model(model_name, data.feature_dim, data.num_classes).to(device)

    partitions = partition_iid(data.y_train, num_clients) if iid else partition_non_iid(
        data.y_train,
        num_clients,
        dirichlet_alpha=dirichlet_alpha,
    )

    val_loader = make_loader(data.X_val, data.y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(data.X_test, data.y_test, batch_size=batch_size, shuffle=False)

    history = {"val_acc": [], "val_f1": [], "test_acc": [], "test_f1": []}

    for _ in range(rounds):
        local_states = []

        for cid in range(num_clients):
            idx = partitions[cid]
            if len(idx) == 0:
                continue

            local_model = deepcopy(global_model).to(device)
            optimizer = optim.AdamW(local_model.parameters(), lr=lr, weight_decay=weight_decay)
            loader = make_loader(data.X_train[idx], data.y_train[idx], batch_size=batch_size, shuffle=True)

            for _ in range(local_epochs):
                train_one_epoch(
                    local_model,
                    loader,
                    optimizer,
                    device,
                    use_dp=use_dp,
                    noise_multiplier=dp_noise_multiplier,
                    max_grad_norm=dp_max_grad_norm,
                )

            local_states.append(OrderedDict((k, v.detach().cpu()) for k, v in local_model.state_dict().items()))

        if not local_states:
            raise RuntimeError("No local updates were produced. Check client partitioning.")

        if use_he:
            aggregated_state = homomorphic_average(local_states)
        elif use_smpc:
            aggregated_state = secure_aggregate_with_masks(local_states)
        else:
            aggregated_state = _average_state_dicts(local_states)

        global_model.load_state_dict(aggregated_state, strict=True)
        global_model.to(device)

        val_metrics = evaluate(global_model, val_loader, device)
        test_metrics = evaluate(global_model, test_loader, device)

        history["val_acc"].append(val_metrics.accuracy)
        history["val_f1"].append(val_metrics.f1_macro)
        history["test_acc"].append(test_metrics.accuracy)
        history["test_f1"].append(test_metrics.f1_macro)

    return history


def run_full_experiment_suite(
    data: DatasetBundle,
    rounds: int = 15,
    num_clients: int = 10,
    local_epochs: int = 2,
    model_name: str = "resmlp",
    device: str = "cuda",
) -> Dict[str, Dict[str, List[float]]]:
    experiments = {
        "fl_iid": dict(iid=True),
        "fl_non_iid": dict(iid=False),
        "fl_smpc": dict(iid=False, use_smpc=True),
        "fl_dp": dict(iid=False, use_dp=True, dp_noise_multiplier=0.6, dp_max_grad_norm=1.0),
        "fl_he": dict(iid=False, use_he=True),
        "fl_hybrid": dict(iid=False, use_smpc=True, use_dp=True, dp_noise_multiplier=0.5, dp_max_grad_norm=1.0),
    }

    results = {}
    for name, kwargs in experiments.items():
        results[name] = run_federated_training(
            data=data,
            model_name=model_name,
            rounds=rounds,
            num_clients=num_clients,
            local_epochs=local_epochs,
            device=device,
            **kwargs,
        )
    return results
