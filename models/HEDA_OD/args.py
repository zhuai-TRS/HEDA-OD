import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.args import add_common_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HEDA-OD: cold cycle residual + hot dual-operator path on OD tensors"
    )
    add_common_args(parser)
    parser.set_defaults(
        model="HEDA_OD",
        loss="mae",
        d_model=256,
        batch_size=32,
        dropout=0.0,
        n_heads=4,
        threshold=3.5,
        attn_dropout=0.5,
        mix_dense_mass_cover=0.80,
        mix_max_dense_channels=2000,
        mix_branch_mode="auto",
        mix_disable_dense_attn=0,
        plot_cycle=1,
        top_k_od=10,
        cycle_gif_fps=4,
    )
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.5,
        help="initial freq threshold for seq/cycle fusion",
    )
    parser.add_argument("--attn_dropout", type=float, default=0.5)
    # ---- ablation / sensitivity knobs (paper Tables) ----
    parser.add_argument(
        "--mix_branch_mode",
        type=str,
        default="auto",
        choices=["auto", "all_sparse", "all_dense"],
        help="auto=mass-cover split; all_sparse=w/o heavy; all_dense=full dense (OOM check)",
    )
    parser.add_argument(
        "--mix_dense_mass_cover",
        type=float,
        default=0.80,
        help="alpha: hot = top activity covering this fraction of raw train mass",
    )
    parser.add_argument(
        "--mix_max_dense_channels",
        type=int,
        default=2000,
        help="hard channel budget C_max",
    )
    parser.add_argument(
        "--mix_disable_dense_attn",
        type=int,
        default=0,
        help="1: w/o attn ablation (freq-fuse+MLP only on hot set)",
    )
    parser.add_argument(
        "--plot_cycle",
        type=int,
        default=1,
        help="1: save cycle_query Top-K heatmap + phase GIF after evaluate",
    )
    parser.add_argument(
        "--top_k_od",
        type=int,
        default=10,
        help="cycle_query heatmap: Top-K active OD pairs from train split",
    )
    parser.add_argument(
        "--cycle_gif_fps",
        type=int,
        default=4,
        help="cycle_query phase GIF frames per second",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def build_setting(args) -> str:
    cmax = int(getattr(args, "mix_max_dense_channels", 2000))
    mode = str(getattr(args, "mix_branch_mode", "auto"))
    alpha = float(getattr(args, "mix_dense_mass_cover", 0.80))
    budget_tag = f"alpha{alpha:.2f}".replace(".", "p")
    noattn = "_noattn" if int(getattr(args, "mix_disable_dense_attn", 0)) else ""
    return (
        f"{args.model_id}_{args.model}_sl{args.seq_len}_pl{args.pred_len}"
        f"_{mode}_{budget_tag}_cmax{cmax}_spnone{noattn}_seed{args.random_seed}"
    )
