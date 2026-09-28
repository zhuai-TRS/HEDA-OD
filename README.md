# HEDA-OD

PyTorch code for **HEDA-OD** (Hot-Pair Enhanced Dual-Operator Allocation for OD Matrix Forecasting).

## Setup

```bash
pip install -r requirements.txt
# Use a CUDA build of PyTorch if needed.
```

## Data

Put `.npy` tensors in `./dataset/` (`--root_path ./dataset/`).

| Dataset | Files | Notes |
|---------|--------|--------|
| Zone OD | `zone_od_matrix.npy` | Private; not released |
| NYC24 Q1–Q3 | `nyc24q1.npy`, `nyc24q2.npy`, `nyc24q3.npy` | [Baidu](https://pan.baidu.com/s/1F2FhFhainZLtAhfQRc_2OQ?pwd=7pgx) code `7pgx` (`HEDA-OD_dataset`) |

## Run

```bash
sh run.sh                      # NYC Q1–Q3
sh scripts/HEDA_OD/nyc24q1.sh  # one quarter
sh scripts/Ablation/zone_branch.sh
sh scripts/Ablation/alpha_grid_zone.sh
```

## Layout

```
run.py / run.sh
models/HEDA_OD/
common/
scripts/HEDA_OD/
scripts/Ablation/
dataset/   # gitignored
```

## Citation

```bibtex
@inproceedings{hedaod2026,
  title     = {{HEDA-OD}: Hot-Pair Enhanced Dual-Operator Allocation for Origin--Destination Matrix Forecasting},
  author    = {Anonymous},
  booktitle = {ISPA},
  year      = {2026}
}
```
