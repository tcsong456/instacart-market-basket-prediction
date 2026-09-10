import logging
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from instacart_rnn.models.model_registry import (
    ModelSpec,
    get_model_spec,
)
from instacart_rnn.training.checkpoint import load_checkpoint
from instacart_rnn.training.export import write_inference_parquet
from instacart_rnn.training.trainer import (
    Trainer,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRunConfig:
    """Configuration required to execute one model training run."""

    model_name: str

    train_path: str
    validation_path: str
    checkpoint_path: str

    epochs: int
    batch_size: int
    read_batch_size: int

    learning_rate: float
    weight_decay: float = 0.0

    num_workers: int = 0
    seed: int = 42

    amp: bool = False
    grad_clip_norm: float | None = None
    early_stopping: int | None = None

    warm_start: bool = False

    pin_memory: bool = False


@dataclass(frozen=True)
class InferenceRunConfig:
    """Configuration required for one inference export run."""

    model_name: str
    eval_path: str
    checkpoint_path: str
    output_path: str

    batch_size: int = 512
    read_batch_size: int = 4096
    rows_per_write: int = 100000

    num_workers: int = 0
    pin_memory: bool = False
    amp: bool = False


def resolve_device() -> torch.device:
    """Return the preferred device for training."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(
    *,
    model: nn.Module,
    spec: ModelSpec,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    return spec.optimizer_factory(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def run_training(
    config: TrainingRunConfig,
) -> None:
    """
    Execute one complete model training process.

    The model registry supplies model-specific construction, losses,
    optimizer configuration and dataloaders.
    """
    logger.info(
        "Starting training run for model %s",
        config.model_name,
    )

    set_random_seed(config.seed)

    device = resolve_device()

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

        logger.info(
            "CUDA device: %s",
            torch.cuda.get_device_name(device),
        )

    spec = get_model_spec(config.model_name)

    logger.info(
        "Building model %s",
        config.model_name,
    )

    model = spec.model_factory()
    model.to(device)

    train_loss_fn = spec.train_loss_factory()
    validation_loss_fn = spec.validation_loss_factory()

    optimizer = _build_optimizer(
        model=model,
        spec=spec,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    logger.info(
        "Building training dataloader from %s",
        config.train_path,
    )

    train_dataloader = spec.dataloader_factory(
        path=config.train_path,
        batch_size=config.batch_size,
        read_batch_size=config.read_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    logger.info(
        "Building validation dataloader from %s",
        config.validation_path,
    )

    validation_dataloader = spec.dataloader_factory(
        path=config.validation_path,
        batch_size=config.batch_size,
        read_batch_size=config.read_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    train_num_rows = spec.count_rows(config.train_path)

    validation_num_rows = spec.count_rows(config.validation_path)

    logger.info(
        "Training rows: %d",
        train_num_rows,
    )

    logger.info(
        "Validation rows: %d",
        validation_num_rows,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loss_fn=train_loss_fn,
        val_loss_fn=validation_loss_fn,
        device=device,
        epochs=config.epochs,
        checkpoint_path=config.checkpoint_path,
        grad_clip_norm=config.grad_clip_norm,
        amp=config.amp,
        early_stopping=config.early_stopping,
    )

    logger.info(
        ("Training %s for %d configured epochs"),
        type(model).__name__,
        config.epochs,
    )

    trainer.fit(
        train_num_rows=train_num_rows,
        val_num_rows=validation_num_rows,
        train_dataloader=train_dataloader,
        val_dataloader=validation_dataloader,
        warm_start=config.warm_start,
    )

    logger.info(
        "Training completed for model %s",
        config.model_name,
    )


def run_inference(
    config: InferenceRunConfig,
) -> None:
    """Run model inference and export representations to Parquet."""

    logger.info(
        "Starting inference run for model %s",
        config.model_name,
    )

    device = resolve_device()

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    spec = get_model_spec(config.model_name)

    logger.info(
        "Building model %s",
        config.model_name,
    )

    model = spec.model_factory()
    model.to(device)

    logger.info(
        "Loading checkpoint from %s",
        config.checkpoint_path,
    )

    load_checkpoint(
        path=config.checkpoint_path,
        model=model,
        device=device,
    )

    logger.info(
        "Building evaluation dataloader from %s",
        config.eval_path,
    )

    dataloader = spec.dataloader_factory(
        path=config.eval_path,
        batch_size=config.batch_size,
        read_batch_size=config.read_batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    num_rows = spec.count_rows(config.eval_path)

    logger.info(
        "Evaluation rows: %d",
        num_rows,
    )

    with tqdm(
        total=num_rows,
        unit="rows",
        desc=f"{model.__class__.__name__} doing inference",
        ncols=100,
        unit_scale=True,
    ) as progress:
        write_inference_parquet(
            model=model,
            dataloader=dataloader,
            device=device,
            output_path=config.output_path,
            rows_per_write=config.rows_per_write,
            amp=config.amp,
            progress=progress,
            batch_tensor_names=spec.inference.batch_tensor_names,
            output_tensor_names=spec.inference.output_tensor_names,
        )

    logger.info(
        "Inference export completed",
    )
