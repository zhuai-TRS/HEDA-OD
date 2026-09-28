"""Train / evaluate deep OD models (loss in scaled space; metrics in original space)."""

import os
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from common.evaluate_od import build_setting, run_phase_evaluation
from common.losses import build_loss
from common.predict_od import collect_predictions


class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss: float, model: nn.Module, path: str):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model, path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model, path)
            self.counter = 0

    def _save(self, val_loss: float, model: nn.Module, path: str):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). "
                "Saving model..."
            )
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss


class ODTrainer:
    def __init__(
        self,
        args,
        model: nn.Module,
        data_provider,
        forward_fn: Optional[Callable] = None,
    ):
        self.args = args
        self.device = self._acquire_device()
        self.model = model.to(self.device)
        self.data_provider = data_provider
        self.forward_fn = forward_fn
        self.criterion = build_loss(args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

    def _acquire_device(self):
        if getattr(self.args, "use_gpu", 1) and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _forward(self, batch):
        if self.forward_fn is not None:
            return self.forward_fn(batch)
        seq_x = batch["seq_x"].float().to(self.device)
        cycle_index = batch["cycle_index"].to(self.device)
        return self.model(seq_x, cycle_index)

    @staticmethod
    def _slice_pred(outputs: torch.Tensor, seq_y: torch.Tensor, pred_len: int):
        return outputs[:, -pred_len:, :, :], seq_y[:, -pred_len:, :, :]

    def _run_epoch(self, loader, train: bool = False) -> float:
        losses = []
        self.model.train() if train else self.model.eval()

        for batch in loader:
            seq_y = batch["seq_y"].float().to(self.device)
            if train:
                self.optimizer.zero_grad()
                outputs = self._forward(batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                pred, target = self._slice_pred(outputs, seq_y, self.args.pred_len)
                loss = self.criterion(pred, target)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self._forward(batch)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    pred, target = self._slice_pred(outputs, seq_y, self.args.pred_len)
                    loss = self.criterion(pred, target)
            losses.append(loss.item())
        return float(np.mean(losses))

    def train(self, setting: str):
        _, train_loader, _ = self.data_provider(self.args, "train")
        _, val_loader, _ = self.data_provider(self.args, "val")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            t0 = time.time()
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            print(
                f"Epoch {epoch + 1}/{self.args.train_epochs} | "
                f"train {train_loss:.6f} | val {val_loss:.6f} | "
                f"time {time.time() - t0:.1f}s"
            )
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        ckpt = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        return path

    def evaluate(self, phase: str = "test", setting: Optional[str] = None):
        _, loader, meta = self.data_provider(self.args, phase)
        scaler = meta.get("scaler") if bool(getattr(self.args, "scale", 1)) else None
        pred, true = collect_predictions(
            self.model,
            loader,
            self.device,
            self.args.pred_len,
            forward_fn=self.forward_fn,
        )
        name = setting or build_setting(self.args)
        return run_phase_evaluation(
            self.args, pred, true, scaler, phase, setting=name
        )
