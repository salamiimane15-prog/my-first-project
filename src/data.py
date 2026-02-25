from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    num_classes: int
    feature_dim: int


def load_ids_dataset(
    path: str,
    target_column: str = "label",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> DatasetBundle:
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    y_raw = df[target_column].astype(str)
    X = df.drop(columns=[target_column])

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X.values,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return DatasetBundle(
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        X_val=X_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        num_classes=len(np.unique(y)),
        feature_dim=X.shape[1],
    )


def partition_iid(y: np.ndarray, num_clients: int, seed: int = 42) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return {cid: split for cid, split in enumerate(splits)}


def partition_non_iid(
    y: np.ndarray,
    num_clients: int,
    dirichlet_alpha: float = 0.3,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in classes:
        idx = class_indices[c]
        rng.shuffle(idx)
        proportions = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
        cut_points = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cut_points)
        for cid, split in enumerate(splits):
            client_indices[cid].extend(split.tolist())

    return {cid: np.array(sorted(indices), dtype=np.int64) for cid, indices in enumerate(client_indices)}


def client_class_distribution(y: np.ndarray, partitions: Dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    for cid, idx in partitions.items():
        labels, counts = np.unique(y[idx], return_counts=True)
        row = {"client_id": cid, "num_samples": len(idx)}
        for label, count in zip(labels, counts):
            row[f"class_{label}"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0).sort_values("client_id")
