# IDS Trend 2025/2026 - Centralized + Federated Privacy-Preserving Pipeline

This repository provides a **professional and modular** codebase for:

1. **Centralized learning** baseline with stronger neural models.
2. **Federated Learning (FL)** with:
   - IID client partitioning
   - Non-IID client partitioning (Dirichlet)
3. **FL + SMPC** (simulated secure aggregation)
4. **FL + DP** (differentially private local training)
5. **FL + Homomorphic Encryption (HE-ready API)**
6. **Hybrid mode** (SMPC + DP)

It is designed to run in **JupyterHub/Jupyter Notebook** and to scale to your hardware (RTX 4090).

---

## Project structure

- `src/data.py`: dataset loading, preprocessing, IID/Non-IID partitioning
- `src/models.py`: `MLPClassifier` and `ResMLPClassifier`
- `src/centralized.py`: centralized training pipeline
- `src/federated.py`: federated simulation and secure variants
- `src/privacy.py`: DP gradient clipping + Gaussian noise
- `src/smpc.py`: simulated secure aggregation
- `src/he.py`: HE integration point (with graceful fallback)
- `scripts/run_pipeline.py`: CLI runner
- `notebooks/ids_2025_2026_pipeline.ipynb`: full notebook workflow

---

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> In JupyterHub: select the kernel from this environment.

---

## 2) Dataset format

Expected format: CSV file where one column is the target label.

Example:
- Features: numeric IDS flow/network features
- Label column: `label` (customizable)

If your label column has another name, pass `target_column` / `--target`.

---

## 3) Quick notebook run (recommended)

Open:

`notebooks/ids_2025_2026_pipeline.ipynb`

Then:
1. Set `DATA_PATH` to your IDS 2025/2026 dataset path.
2. Keep `MODEL_NAME = "resmlp"` for stronger baseline.
3. Run all cells.
4. Compare centralized vs all FL variants in the summary table.

---

## 4) Command-line run

```bash
python scripts/run_pipeline.py \
  --data /path/to/ids_trend_2025_2026.csv \
  --target label \
  --model resmlp \
  --epochs 20 \
  --rounds 15 \
  --num-clients 10 \
  --local-epochs 2 \
  --device cuda \
  --output results.json
```

---

## 5) Recommended hyperparameters for RTX 4090

Start with:
- `model=resmlp`
- `batch_size=1024` (or 2048 if memory allows)
- `epochs=20-40` for centralized baseline
- FL:
  - `num_clients=10-20`
  - `rounds=15-40`
  - `local_epochs=1-3`

For non-IID realism:
- `dirichlet_alpha=0.1 to 0.5` (lower => more heterogeneity)

For DP:
- `noise_multiplier=0.4 to 0.8`
- `max_grad_norm=1.0`

---

## 6) Notes on security modes

- **SMPC** in this repo is a **simulation** of secure aggregation via additive masks.
- **DP** is implemented in local client training (gradient clipping + noise).
- **HE** is provided with an integration-ready API and plaintext fallback to keep experiments runnable even if HE libraries are unavailable.
- **Hybrid** currently combines SMPC + DP.

---

## 7) How to improve accuracy further

1. Better feature engineering (flow time windows, attack-family grouping).
2. Class imbalance handling (class-weighted loss or focal loss).
3. Hyperparameter sweep (`lr`, hidden dim, rounds, local epochs).
4. Early stopping and model checkpointing.
5. Ensemble between centralized and federated teacher-student distillation.

---

## 8) Troubleshooting

- If CUDA is unavailable, the code falls back to CPU.
- If your CSV has non-numeric columns, they are coerced safely.
- Missing values and infinities are sanitized.

