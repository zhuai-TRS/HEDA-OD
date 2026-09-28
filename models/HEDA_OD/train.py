"""Wire HEDA_OD to shared ODTrainer (train in scaled space)."""

import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.channel_split import split_summary
from models.HEDA_OD.trainer import MemorySkipped, HEDAODTrainer
from models.HEDA_OD.args import build_setting, parse_args
from models.HEDA_OD.data import od_data_provider
from models.HEDA_OD.model import Model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args=None):
    args = parse_args() if args is None else args
    set_seed(args.random_seed)

    _, _, meta = od_data_provider(args, "train")
    args.num_nodes = meta["raw_data"].shape[1]

    model = Model(args)
    model.init_channel_split(meta, args)
    split = split_summary(model.sparse_mask.cpu().numpy())
    info = getattr(model, "_split_info", {}) or {}
    print(
        f"HEDA_OD split | mode={args.mix_branch_mode} | "
        f"sparse={split['sparse_channels']} | dense={split['dense_channels']} | "
        f"split_info={info}"
    )

    trainer = HEDAODTrainer(args, model, od_data_provider)
    setting = build_setting(args)

    try:
        if args.is_training:
            print(">>>>>>> HEDA_OD training >>>>>>>")
            trainer.train(setting)
            for phase in ("val", "test"):
                trainer.evaluate(phase=phase, setting=setting)
        else:
            print(">>>>>>> HEDA_OD evaluation only >>>>>>>")
            ckpt = Path(args.checkpoints) / setting / "checkpoint.pth"
            if ckpt.is_file():
                trainer.model.load_state_dict(
                    torch.load(ckpt, map_location=trainer.device)
                )
            for phase in ("val", "test"):
                trainer.evaluate(phase=phase, setting=setting)
    except MemorySkipped as exc:
        print(exc)
        print("Skipping this run.")
        return setting

    return setting
