from __future__ import annotations

import argparse
import json

from src.centralized import run_centralized_training
from src.data import load_ids_dataset
from src.federated import run_full_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="IDS 2025/2026 centralized + federated pipeline")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="label", help="Target column name")
    parser.add_argument("--model", default="resmlp", choices=["mlp", "resmlp"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    data = load_ids_dataset(path=args.data, target_column=args.target)

    centralized = run_centralized_training(
        data=data,
        model_name=args.model,
        epochs=args.epochs,
        device=args.device,
    )

    federated = run_full_experiment_suite(
        data=data,
        rounds=args.rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        model_name=args.model,
        device=args.device,
    )

    payload = {"centralized": centralized, "federated": federated}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
