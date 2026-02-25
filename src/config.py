from dataclasses import dataclass
from typing import Literal


@dataclass
class DataConfig:
    data_path: str
    target_column: str = "label"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


@dataclass
class TrainConfig:
    model_name: Literal["mlp", "resmlp"] = "resmlp"
    batch_size: int = 1024
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"


@dataclass
class FLConfig:
    rounds: int = 15
    num_clients: int = 10
    local_epochs: int = 2
    fraction_fit: float = 1.0
    iid: bool = True
    dirichlet_alpha: float = 0.3


@dataclass
class DPConfig:
    enabled: bool = False
    noise_multiplier: float = 0.8
    max_grad_norm: float = 1.0


@dataclass
class SecurityConfig:
    use_smpc: bool = False
    use_dp: bool = False
    use_he: bool = False
    hybrid_mode: bool = False
