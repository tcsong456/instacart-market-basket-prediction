from collections.abc import Callable, Iterable
from typing import Any

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from tqdm import tqdm

from instacart_etl_rnn.common.paths import join_path

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


def _build_ckpt_name(path: str, model: nn.Module) -> str:
    model_name = model.__class__.__name__.lower()
    ckpt_name = f"best_{model_name}_ckpt.pt"
    return str(join_path(str(path), ckpt_name))


def _get_output_tensor(
    output: Any,
    name: str,
) -> torch.Tensor:
    if isinstance(output, dict):
        value = output[name]
    else:
        value = getattr(output, name)

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Model output {name!r} is not a torch.Tensor")

    return value


def _concatenate_buffer(
    buffer: dict[str, list[torch.Tensor]],
) -> TensorBatch:
    return {name: torch.cat(tensors, dim=0) for name, tensors in buffer.items()}


def _to_arrow_table(
    tensors: TensorBatch,
    batch_tensor_names: tuple[str, ...] = (),
    output_tensor_names: tuple[str, ...] = (),
) -> pa.Table:
    """
    Convert buffered inference tensors into an Arrow table.
    """
    final_states = tensors["final_states"]

    if final_states.ndim != 2:
        raise ValueError(
            "final_states must have shape [rows, hidden_size], "
            f"but received {tuple(final_states.shape)}"
        )

    state_width = final_states.shape[1]

    final_states_array = pa.FixedSizeListArray.from_arrays(
        pa.array(
            final_states.reshape(-1).numpy(),
            type=pa.float32(),
        ),
        list_size=state_width,
    )

    output_table: dict[str, pa.Array] = {}

    for name in batch_tensor_names:
        output_table[name] = pa.array(tensors[name].numpy(), type=pa.int64())

    for name in output_tensor_names:
        if name == "final_states":
            output_table[name] = final_states_array
        elif name == "final_logits":
            output_table[name] = pa.array(
                tensors[name].numpy(),
                type=pa.float32(),
            )
            output_table["final_probabilities"] = pa.array(
                tensors["final_probabilities"].numpy(),
                type=pa.float32(),
            )
        else:
            raise KeyError(
                "Only final_states and final_logits "
                "are supported for output_tensor_names, but received "
                f"{name}"
            )

    return pa.table(output_table)


def _clear_buffer(
    buffer: dict[str, list[torch.Tensor]],
) -> None:
    for tensors in buffer.values():
        tensors.clear()


def _is_gcs_path(path: str) -> bool:
    return str(path).startswith("gs://")


def _open_output_file(output_path: str):
    output_path = str(output_path)

    if _is_gcs_path(output_path):
        return gcsfs.GCSFileSystem().open(output_path, "wb")

    return output_path


def _close_output_file(sink: Any) -> None:
    if hasattr(sink, "close") and not getattr(sink, "closed", False):
        sink.close()


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: Iterable[TensorBatch],
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: Any | None = None,
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

        if scheduler is not None:
            scheduler.step()


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


@torch.no_grad()
def iter_inference(
    *,
    model: nn.Module,
    dataloader: Iterable[TensorBatch],
    device: torch.device,
    amp: bool = False,
    batch_tensor_names: tuple[str, ...] = (),
    output_tensor_names: tuple[str, ...] = (),
):
    """
    Yield CPU inference tensors one model batch at a time.

    Args:
        model: Model used for inference.
        dataloader: Iterable yielding model-ready tensor batches.
        device: Device used for model inference.
        amp: Whether to enable CUDA automatic mixed precision.

    Yields:
        CPU tensors required for downstream stacking.
    """
    model.eval()

    amp_enabled = amp and device.type == "cuda"

    collected: TensorBatch = {}

    for batch in dataloader:
        batch = move_batch_to_device(
            batch,
            device,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            output = model(batch)

        for name in batch_tensor_names:
            collected[name] = batch[name].detach().cpu()

        for name in output_tensor_names:
            collected[name] = (
                _get_output_tensor(
                    output,
                    name,
                )
                .detach()
                .cpu()
            )

        collected["final_probabilities"] = torch.sigmoid(collected["final_logits"])

        yield collected


def write_inference_parquet(
    *,
    model: nn.Module,
    dataloader: Iterable[TensorBatch],
    device: torch.device,
    output_path: str,
    amp: bool = False,
    rows_per_write: int = 100_000,
    compression: str = "snappy",
    batch_tensor_names: tuple[str, ...] = (),
    output_tensor_names: tuple[str, ...] = (),
    progress: Any | None = None,
) -> None:
    """
    Run inference and stream stacking features into a Parquet file.

    Model batches are accumulated on CPU until `rows_per_write` is reached,
    then written as a Parquet row group.

    Args:
        model: Model used for inference.
        dataloader: Iterable yielding model-ready tensor batches.
        device: Device used for inference.
        output_path: Local path or ``gs://`` URL of the destination Parquet file.
        amp: Whether to enable CUDA automatic mixed precision.
        rows_per_write: Approximate number of rows buffered before each write.
        compression: Parquet compression codec.
    """
    if rows_per_write <= 0:
        raise ValueError("rows_per_write must be greater than 0")

    buffer: dict[str, list[torch.Tensor]] | None = None
    buffered_rows = 0
    writer: pq.ParquetWriter | None = None
    sink: Any = None

    try:
        for inference_batch in iter_inference(
            model=model,
            dataloader=dataloader,
            device=device,
            amp=amp,
            batch_tensor_names=batch_tensor_names,
            output_tensor_names=output_tensor_names,
        ):
            if buffer is None:
                buffer = {name: [] for name in inference_batch}

            batch_rows = next(iter(inference_batch.values())).shape[0]

            for name, tensor in inference_batch.items():
                buffer[name].append(tensor)

            buffered_rows += batch_rows

            if buffered_rows < rows_per_write:
                continue

            tensors = _concatenate_buffer(buffer)
            table = _to_arrow_table(
                tensors,
                batch_tensor_names=batch_tensor_names,
                output_tensor_names=output_tensor_names,
            )

            if writer is None:
                sink = _open_output_file(output_path)
                writer = pq.ParquetWriter(
                    where=sink,
                    schema=table.schema,
                    compression=compression,
                )

            writer.write_table(table)

            if progress is not None:
                progress.update(buffered_rows)

            _clear_buffer(buffer)
            buffered_rows = 0

        if buffered_rows > 0 and buffer is not None:
            tensors = _concatenate_buffer(buffer)
            table = _to_arrow_table(
                tensors,
                batch_tensor_names=batch_tensor_names,
                output_tensor_names=output_tensor_names,
            )

            if writer is None:
                sink = _open_output_file(output_path)
                writer = pq.ParquetWriter(
                    where=sink,
                    schema=table.schema,
                    compression=compression,
                )

            writer.write_table(table)

            if progress is not None:
                progress.update(buffered_rows)
    finally:
        if writer is not None:
            writer.close()

        _close_output_file(sink)


def save_checkpoint(
    *,
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    scheduler: Any | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "validation_loss": validation_loss,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    path = str(path)
    ckpt_path = _build_ckpt_name(path, model)

    if _is_gcs_path(path):
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
) -> dict[str, Any]:
    path = str(path)
    ckpt_path = _build_ckpt_name(path, model)

    if _is_gcs_path(path):
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

    return checkpoint


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
        scheduler: Any | None = None,
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
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(amp and device.type == "cuda"),
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
                    scheduler=self.scheduler,
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

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_epoch = epoch
                bad_epochs = 0

                save_checkpoint(
                    path=self.checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=best_epoch,
                    validation_loss=(best_validation_loss),
                )
            else:
                bad_epochs += 1

            if self.early_stopping is not None and bad_epochs >= self.early_stopping:
                print(f"early stopping at epoch={epoch}")
                break

    def inference(
        self,
        *,
        output_path: str,
        eval_num_rows: int,
        rows_per_write: int,
        dataloader: Iterable[TensorBatch],
        batch_tensor_names: tuple[str, ...] = (),
        output_tensor_names: tuple[str, ...] = (),
    ) -> None:
        load_checkpoint(
            path=self.checkpoint_path,
            model=self.model,
            device=self.device,
        )

        with tqdm(
            total=eval_num_rows,
            unit="rows",
            desc=f"{self.model.__class__.__name__} doing inference",
            ncols=100,
            unit_scale=True,
        ) as progress:
            write_inference_parquet(
                model=self.model,
                dataloader=dataloader,
                device=self.device,
                output_path=output_path,
                amp=self.amp,
                rows_per_write=rows_per_write,
                batch_tensor_names=batch_tensor_names,
                output_tensor_names=output_tensor_names,
                progress=progress,
            )
