from typing import Any

import gcsfs
import torch
from torch import nn

from instacart_etl_rnn.common.paths import is_gcs_url, join_path


def _build_ckpt_name(path: str, model: nn.Module) -> str:
    model_name = model.__class__.__name__.lower()
    ckpt_name = f"best_{model_name}_ckpt.pt"
    return str(join_path(str(path), ckpt_name))


def save_checkpoint(
    *,
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
    }

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    path = str(path)
    ckpt_path = _build_ckpt_name(path, model)

    if is_gcs_url(path):
        fs = gcsfs.GCSFileSystem()

        with fs.open(ckpt_path, "wb") as f:
            torch.save(checkpoint, f)

        return

    torch.save(checkpoint, ckpt_path)


def load_checkpoint(
    *,
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    path = str(path)
    ckpt_path = _build_ckpt_name(path, model)

    if is_gcs_url(path):
        fs = gcsfs.GCSFileSystem()

        with fs.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(
                f,
                map_location=device,
            )
    else:
        checkpoint = torch.load(
            ckpt_path,
            map_location=device,
        )

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint
