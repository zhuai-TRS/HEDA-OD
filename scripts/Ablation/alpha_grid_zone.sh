#!/usr/bin/env bash
# Alpha mass-cover grid on zone OD at H=8 (paper Table for alpha sensitivity).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

for alpha in 0.50 0.60 0.70 0.80 0.90; do
  python -u run.py \
    --root_path ./dataset/ \
    --data_path zone_od_matrix.npy \
    --model_id "zone_alpha_${alpha}" \
    --model HEDA_OD \
    --seq_len 12 --pred_len 8 --cycle_len 96 \
    --d_model 256 --n_heads 4 \
    --mix_branch_mode auto \
    --mix_dense_mass_cover "${alpha}" \
    --mix_max_dense_channels 2000 \
    --dropout 0 \
    --attn_dropout 0.5 \
    --batch_size 32 --learning_rate 0.001 \
    --train_epochs 30 --patience 5 \
    --loss mae --scale 1 --is_training 1 --random_seed 2026 \
    --plot_od_pairs 0 --plot_cycle 0
done
