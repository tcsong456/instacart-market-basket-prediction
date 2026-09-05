import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

PRODUCT_TRAINING_COLUMNS: tuple[str, ...] = (
    "user_id",
    "product_id",
    "aisle_id",
    "department_id",
    "label",
    "product_name_encoded",
    "is_ordered_history",
    "position_in_order_history",
    "history_order_size",
    "history_reorder_size",
    "order_dows",
    "order_hours",
    "days_since_prior_orders",
    "order_numbers",
    "history_length",
    "product_name_length",
)

BatchDict = dict[str, torch.Tensor]


def _shift_left(x: torch.Tensor) -> torch.Tensor:
    """Drop the first timestep and pad a trailing zero (next-step alignment)."""

    pad = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
    return torch.cat([x[:, 1:], pad], dim=1)


def _scalar_numpy(column: pa.Array, dtype: np.dtype) -> np.ndarray:
    """Convert an Arrow scalar column to a writable NumPy array."""

    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)


def _fixed_list_numpy(column: pa.Array, dtype: np.dtype, width: int) -> np.ndarray:
    n_rows = len(column)

    offset = column.offset
    values = np.asarray(
        column.values.slice(offset * width, n_rows * width).to_numpy(
            zero_copy_only=False
        ),
        dtype=dtype,
    )
    return values.reshape(n_rows, width)


def _as_tensor(values: np.ndarray) -> torch.Tensor:
    """Wrap a NumPy array as a tensor without an extra copy when possible."""

    if not values.flags.writeable:
        values = np.array(values, copy=True)
    return torch.from_numpy(values)


def _worker_shard() -> tuple[int, int]:
    """Return ``(global_worker_id, global_num_workers)`` for DataLoader + DDP."""

    worker_info = get_worker_info()
    if worker_info is None:
        worker_id, num_workers = 0, 1
    else:
        worker_id, num_workers = worker_info.id, worker_info.num_workers

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1

    return rank * num_workers + worker_id, world_size * num_workers


def _permute_batch(batch: BatchDict, generator: torch.Generator) -> BatchDict:
    """Shuffle rows inside one tensor batch with a seeded generator."""

    n = next(iter(batch.values())).shape[0]
    if n <= 1:
        return batch
    order = torch.randperm(n, generator=generator)
    return {name: tensor[order] for name, tensor in batch.items()}


def _concat_batches(left: BatchDict, right: BatchDict) -> BatchDict:
    return {name: torch.cat([left[name], right[name]], dim=0) for name in left}


class BaseIterableDataset(IterableDataset, ABC):
    """Iterable Parquet dataset that yields training mini-batches.

    Prefer multi-file Parquet directories so fragment sharding can keep workers busy.
    Call :meth:`set_epoch` at the start of every epoch when shuffle is enabled.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        read_batch_size: int = 4096,
        batch_size: int = 64,
        drop_last: bool = True,
        shuffle_fragments: bool = True,
        shuffle_rows: bool = True,
        seed: int = 42,
        columns: tuple[str, ...] = PRODUCT_TRAINING_COLUMNS,
    ) -> None:
        super().__init__()

        if read_batch_size <= 0:
            raise ValueError("read_batch_size must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.path = str(path)
        self.read_batch_size = read_batch_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_fragments = shuffle_fragments
        self.shuffle_rows = shuffle_rows
        self.seed = seed
        self.columns = columns
        self.epoch = 0

        self._validate_schema()

    @abstractmethod
    def record_batch_to_features(self, record_batch: pa.RecordBatch) -> BatchDict:
        pass

    def _validate_schema(self) -> None:
        """Fail fast if required columns are missing from the Parquet dataset."""

        dataset = ds.dataset(self.path, format="parquet")
        names = set(dataset.schema.names)
        missing = [name for name in self.columns if name not in names]
        if missing:
            raise ValueError(
                f"Parquet dataset at {self.path!r} is missing columns: {missing}"
            )

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch so fragment/row shuffle order changes."""

        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        self.epoch = epoch

    def __iter__(self) -> Iterator[BatchDict]:
        dataset = ds.dataset(self.path, format="parquet")
        fragments = list(dataset.get_fragments())

        worker_id, num_workers = _worker_shard()

        if self.shuffle_fragments:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(fragments)

        fragments = fragments[worker_id::num_workers]

        pending: BatchDict | None = None

        for fragment_index, fragment in enumerate(fragments):
            scanner = fragment.scanner(
                columns=list(self.columns),
                batch_size=self.read_batch_size,
                use_threads=False,
            )

            for batch_index, record_batch in enumerate(scanner.to_batches()):
                if record_batch.num_rows == 0:
                    continue

                features = self.record_batch_to_features(record_batch)

                if self.shuffle_rows:
                    generator = torch.Generator()
                    generator.manual_seed(
                        self.seed
                        + self.epoch * 1_000_003
                        + worker_id * 10_007
                        + fragment_index * 97
                        + batch_index
                    )
                    features = _permute_batch(features, generator)

                if pending is not None:
                    features = _concat_batches(pending, features)
                    pending = None

                row_count = features["user_id"].shape[0]
                full_end = (row_count // self.batch_size) * self.batch_size

                for start in range(0, full_end, self.batch_size):
                    end = start + self.batch_size
                    yield {name: tensor[start:end] for name, tensor in features.items()}

                if full_end < row_count:
                    pending = {
                        name: tensor[full_end:] for name, tensor in features.items()
                    }

        if pending is not None and not self.drop_last:
            yield pending


class ProductIterableDataset(BaseIterableDataset):
    def record_batch_to_features(self, record_batch: pa.RecordBatch) -> BatchDict:
        """Map one Arrow record batch into model feature tensors.

        Applies the peel used by the product RNN:
        - temporal / order-state sequences are left-shifted;
        - ``history_length`` is preserved for next-basket prediction;
        - ``sequence_loss_length`` is ``history_length - 1``;
        - ``is_none`` marks the synthetic product_id == 0 row.
        """

        def scalar(name: str, dtype: np.dtype) -> torch.Tensor:
            return _as_tensor(_scalar_numpy(record_batch.column(name), dtype))

        def sequence(name: str, dtype: np.dtype, width: int) -> torch.Tensor:
            return _as_tensor(
                _fixed_list_numpy(record_batch.column(name), dtype, width)
            )

        is_ordered_history = sequence("is_ordered_history", np.int64, 100)

        batch: BatchDict = {
            "user_id": scalar("user_id", np.int64),
            "product_id": scalar("product_id", np.int64),
            "aisle_id": scalar("aisle_id", np.int64),
            "department_id": scalar("department_id", np.int64),
            "label": scalar("label", np.int64),
            "product_name_encoded": sequence("product_name_encoded", np.int64, 30),
            "position_in_order_history": sequence(
                "position_in_order_history", np.int64, 100
            ),
            "history_order_size": sequence("history_order_size", np.int64, 100),
            "history_reorder_size": sequence("history_reorder_size", np.int64, 100),
            "is_ordered_history": is_ordered_history,
            "order_dow_history": _shift_left(sequence("order_dows", np.int64, 100)),
            "order_hour_history": _shift_left(sequence("order_hours", np.int64, 100)),
            "days_since_prior_order_history": _shift_left(
                sequence("days_since_prior_orders", np.int64, 100)
            ),
            "order_number_history": _shift_left(
                sequence("order_numbers", np.int64, 100)
            ),
            "next_is_ordered": _shift_left(is_ordered_history),
            "history_length": scalar("history_length", np.int64),
            "sequence_loss_length": scalar("history_length", np.int64) - 1,
            "product_name_length": scalar("product_name_length", np.int64),
        }
        batch["is_none"] = batch["product_id"] == 0
        return batch


def create_product_dataloader(
    path: str | Path,
    *,
    batch_size: int = 64,
    read_batch_size: int = 4096,
    num_workers: int = 0,
    drop_last: bool = True,
    shuffle: bool = True,
    seed: int = 42,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    """Build a DataLoader that streams product-training Parquet batches.

    The underlying dataset already emits mini-batches, so this sets
    ``batch_size=None``. For training, call ``loader.dataset.set_epoch(epoch)``
    each epoch when ``shuffle=True``.
    """

    dataset = ProductIterableDataset(
        path,
        read_batch_size=read_batch_size,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle_fragments=shuffle,
        shuffle_rows=shuffle,
        seed=seed,
    )

    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)
