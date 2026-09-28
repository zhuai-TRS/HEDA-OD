#!/usr/bin/env bash
# Ablations: w/o heavy (all_sparse) and w/o channel attention (noattn).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

run_one() {
  local tag=$1; shift
  for pred_len in 4 8 12; do
    python -u run.py \
      --root_path ./dataset/ \
      --data_path zone_od_matrix.npy \
      --model_id "zone_abl_${tag}" \
      --model HEDA_OD \
      --seq_len 12 --pred_len "${pred_len}" --cycle_len 96 \
      --d_model 256 --n_heads 4 \
      --mix_dense_mass_cover 0.80 \
      --mix_max_dense_channels 2000 \
      --dropout 0 \
      --attn_dropout 0.5 \
      --batch_size 32 --learning_rate 0.001 \
      --train_epochs 30 --patience 5 \
      --loss mae --scale 1 --is_training 1 --random_seed 2026 \
      --plot_od_pairs 0 --plot_cycle 0 \
      "$@"
  done
}

run_one all_sparse --mix_branch_mode all_sparse
run_one noattn --mix_disable_dense_attn 1
