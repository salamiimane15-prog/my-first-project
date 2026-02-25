from __future__ import annotations

from typing import Dict, List

import torch
from torch import optim

from src.data import DatasetBundle
from src.models import build_model
from src.train_utils import evaluate, make_loader, train_one_epoch


def run_centralized_training(
    data: DatasetBundle,
    model_name: str = "resmlp",
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
) -> Dict[str, List[float]]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = build_model(model_name, data.feature_dim, data.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_loader(data.X_train, data.y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(data.X_val, data.y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(data.X_test, data.y_test, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for _ in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics.loss)
        history["val_acc"].append(val_metrics.accuracy)
        history["val_f1"].append(val_metrics.f1_macro)

    test_metrics = evaluate(model, test_loader, device)
    history["test_loss"] = [test_metrics.loss]
    history["test_acc"] = [test_metrics.accuracy]
    history["test_f1"] = [test_metrics.f1_macro]

    return history
