"""Shared CLI arguments for OD forecasting experiments."""

import argparse


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_id", type=str, default="zone_od")
    parser.add_argument("--model", type=str, default="HEDA_OD")

    parser.add_argument("--root_path", type=str, default="./dataset/")
    parser.add_argument("--data_path", type=str, default="zone_od_matrix.npy")
    parser.add_argument("--seq_len", type=int, default=12, help="history window X")
    parser.add_argument("--pred_len", type=int, default=12, help="forecast horizon Y")
    parser.add_argument("--label_len", type=int, default=0)

    parser.add_argument(
        "--cycle_len",
        type=int,
        default=96,
        help="one calendar day in slots: 96 for 15-min zone, 24 for hourly NYC",
    )
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--split_lens",
        type=str,
        default="",
        help="optional absolute train,val,test lengths, e.g. 1246,142,349 "
        "(overrides ratios)",
    )
    parser.add_argument("--scale", type=int, default=1, help="1: log1p+z-score; 0: raw counts")
    parser.add_argument("--scale_eps", type=float, default=1e-8)
    parser.add_argument(
        "--scale_std_floor",
        type=float,
        default=1e-2,
        help="min per-OD std in log1p space when scale=1",
    )
    parser.add_argument(
        "--loss_z_clip",
        type=float,
        default=50.0,
        help="clamp |pred|/|true| in loss when scale=1; 0=disable",
    )
    parser.add_argument(
        "--mape_min_true",
        type=float,
        default=1.0,
        help="MAPE / nonzero metrics: true >= this (original OD counts)",
    )

    parser.add_argument("--is_training", type=int, default=1, help="1: train+test; 0: eval only")
    parser.add_argument("--d_model", type=int, default=256, help="hidden dim")
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--loss",
        type=str,
        default="mae",
        choices=["mse", "mae", "masked_mse", "sparse_mae", "nonzero_mae"],
    )
    parser.add_argument("--nonzero_weight", type=float, default=5.0)
    parser.add_argument("--zero_weight", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--results", type=str, default="./results/")
    parser.add_argument(
        "--plot_od_pairs",
        type=int,
        default=0,
        help="1: save pred/true line plots for selected OD pairs",
    )
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--random_seed", type=int, default=2026)
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    return parser


def get_common_parser() -> argparse.ArgumentParser:
    return add_common_args(
        argparse.ArgumentParser(description="HEDA-OD OD forecasting")
    )
