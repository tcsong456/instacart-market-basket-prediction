from collections.abc import Iterable
from typing import Any

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

from instacart_etl_rnn.common.paths import is_gcs_url, join_path
from instacart_rnn.training.trainer import TensorBatch, move_batch_to_device


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


def _open_output_file(output_path: str):
    output_path = str(output_path)

    if is_gcs_url(output_path):
        return gcsfs.GCSFileSystem().open(output_path, "wb")

    return output_path


def _close_output_file(sink: Any) -> None:
    if hasattr(sink, "close") and not getattr(sink, "closed", False):
        sink.close()


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

    output_path = join_path(
        output_path, f"{model.__class__.__name__.lower()}_representation.parquet"
    )

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
