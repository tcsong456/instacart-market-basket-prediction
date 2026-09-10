from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from instacart_rnn.training.checkpoint import load_checkpoint, save_checkpoint

TensorBatch = dict[str, torch.Tensor]

LossFn = Callable[
    [Any, TensorBatch],
    torch.Tensor,
]


def move_batch_to_device(
    batch: TensorBatch,
    device: torch.device,
) -> TensorBatch:
    return {
        name: tensor.to(
            device,
            non_blocking=True,
        )
        for name, tensor in batch.items()
    }


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: Iterable[TensorBatch],
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip_norm: float | None = None,
    amp: bool = False,
    progress: Any | None = None,
) -> None:
    model.train()

    amp_enabled = amp and device.type == "cuda"

    total_loss = 0.0
    total_rows = 0

    for batch in dataloader:
        current_lr = optimizer.param_groups[0]["lr"]

        batch = move_batch_to_device(
            batch,
            device,
        )

        batch_size = batch["user_id"].size(0)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            output = model(batch)

            loss = loss_fn(
                output,
                batch,
            )

        if progress is not None:
            valid_count = batch["sequence_loss_length"].sum().item()

            total_loss += loss.item() * valid_count
            total_rows += valid_count

            running_loss = total_loss / total_rows

            progress.update(batch_size)
            progress.set_postfix(loss=f"{running_loss:.5f}", lr=f"{current_lr: 2f}")

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )

            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )

            optimizer.step()


@torch.no_grad()
def evaluate(
    *,
    model: nn.Module,
    dataloader: Iterable[TensorBatch],
    loss_fn: LossFn,
    device: torch.device,
    amp: bool = False,
    progress: Any | None = None,
) -> float:
    model.eval()

    amp_enabled = amp and device.type == "cuda"

    total_loss = 0.0
    total_rows = 0

    for batch in dataloader:
        batch = move_batch_to_device(
            batch,
            device,
        )

        batch_size = batch["user_id"].size(0)

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            output = model(batch)

            loss = loss_fn(
                output,
                batch,
            )

        total_loss += loss.item() * batch_size
        total_rows += batch_size

        if progress is not None:
            running_loss = total_loss / max(total_rows, 1)

            progress.update(batch_size)
            progress.set_postfix(
                loss=f"{running_loss:.5f}",
            )

    if total_rows == 0:
        return 0.0

    return round(total_loss / total_rows, 5)


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
        checkpoint_path: str,
        device: torch.device,
        epochs: int,
        early_stopping: int | None = None,
        grad_clip_norm: float | None = None,
        amp: bool = False,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer

        self.train_loss_fn = train_loss_fn
        self.val_loss_fn = val_loss_fn

        self.device = device

        self.epochs = epochs
        self.early_stopping = early_stopping
        self.grad_clip_norm = grad_clip_norm
        self.amp = amp
        self.checkpoint_path = checkpoint_path

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(amp and device.type == "cuda"),
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=1,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=0,
            min_lr=1e-6,
        )

    def fit(
        self,
        *,
        train_num_rows: int,
        val_num_rows: int,
        train_dataloader: Iterable[TensorBatch],
        val_dataloader: Iterable[TensorBatch],
        warm_start: bool = False,
    ) -> None:
        start_epoch = 0
        best_epoch = -1
        best_validation_loss = float("inf")
        bad_epochs = 0

        if warm_start:
            checkpoint = load_checkpoint(
                path=self.checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
                scaler=self.scaler,
            )

            start_epoch = checkpoint["epoch"] + 1

            best_validation_loss = checkpoint["validation_loss"]

        for epoch in range(
            start_epoch,
            self.epochs,
        ):
            with tqdm(
                total=train_num_rows,
                unit="rows",
                desc=f"{self.model.__class__.__name__} training epoch {epoch}",
                ncols=100,
                unit_scale=True,
            ) as progress:
                train_one_epoch(
                    model=self.model,
                    dataloader=train_dataloader,
                    optimizer=self.optimizer,
                    loss_fn=self.train_loss_fn,
                    device=self.device,
                    scaler=self.scaler,
                    grad_clip_norm=(self.grad_clip_norm),
                    amp=self.amp,
                    progress=progress,
                )

            with tqdm(
                total=val_num_rows,
                unit="rows",
                desc=f"{self.model.__class__.__name__} evaluating epoch {epoch}",
                ncols=100,
                unit_scale=True,
            ) as progress:
                val_loss = evaluate(
                    model=self.model,
                    dataloader=val_dataloader,
                    loss_fn=self.val_loss_fn,
                    device=self.device,
                    amp=self.amp,
                    progress=progress,
                )

            self.scheduler.step(val_loss)

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_epoch = epoch
                bad_epochs = 0

                save_checkpoint(
                    path=self.checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=best_epoch,
                    validation_loss=(best_validation_loss),
                )
            else:
                bad_epochs += 1

            if self.early_stopping is not None and bad_epochs >= self.early_stopping:
                print(f"early stopping at epoch={epoch}")
                break
