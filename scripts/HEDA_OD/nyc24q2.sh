#!/usr/bin/env bash
# HEDA-OD main experiment (seed 2026).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

data_path=nyc24q2.npy
model_id=nyc24q2
seq_len=12
cycle_len=24
d_model=256
batch_size=16
alpha=0.80
cmax=2000

for pred_len in 4 8 12; do
  python -u run.py \
    --root_path ./dataset/ \
    --data_path "${data_path}" \
    --model_id "${model_id}" \
    --model HEDA_OD \
    --seq_len "${seq_len}" \
    --pred_len "${pred_len}" \
    --cycle_len "${cycle_len}" \
    --d_model "${d_model}" \
    --n_heads 4 \
    --mix_branch_mode auto \
    --mix_dense_mass_cover "${alpha}" \
    --mix_max_dense_channels "${cmax}" \
    --dropout 0 \
    --attn_dropout 0.5 \
    --batch_size "${batch_size}" \
    --learning_rate 0.001 \
    --train_epochs 30 \
    --patience 5 \
    --loss mae \
    --scale 1 \
    --is_training 1 \
    --random_seed 2026 \
    --plot_od_pairs 0 \
    --plot_cycle 0
done
