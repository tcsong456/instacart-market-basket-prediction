from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from instacart_rnn.dataset import (
    PRODUCT_TRAINING_COLUMNS,
    ProductIterableDataset,
    _worker_shard,
    create_product_dataloader,
)


def _fixed_list_array(rows, value_type):
    width = len(rows[0])
    return pa.array(rows, type=pa.list_(value_type, width))


def _training_table(user_ids):
    row_count = len(user_ids)
    history_width = 100
    product_name_width = 30

    return pa.table(
        {
            "user_id": pa.array(user_ids, type=pa.int32()),
            "product_id": pa.array(
                [0 if user_id == user_ids[0] else user_id + 10 for user_id in user_ids],
                type=pa.int32(),
            ),
            "aisle_id": pa.array([1] * row_count, type=pa.int32()),
            "department_id": pa.array([2] * row_count, type=pa.int32()),
            "label": pa.array(
                [user_id % 2 for user_id in user_ids],
                type=pa.int32(),
            ),
            "product_name_encoded": _fixed_list_array(
                [[7] * product_name_width for _ in user_ids],
                pa.int32(),
            ),
            "is_ordered_history": _fixed_list_array(
                [[1, 0, 1] + [0] * (history_width - 3) for _ in user_ids],
                pa.int32(),
            ),
            "position_in_order_history": _fixed_list_array(
                [[1, 0, 2] + [0] * (history_width - 3) for _ in user_ids],
                pa.int32(),
            ),
            "history_order_size": _fixed_list_array(
                [[2, 3, 4] + [0] * (history_width - 3) for _ in user_ids],
                pa.int32(),
            ),
            "history_reorder_size": _fixed_list_array(
                [[0, 1, 2] + [0] * (history_width - 3) for _ in user_ids],
                pa.int32(),
            ),
            "order_dows": _fixed_list_array(
                [[0, 2, 4, 6] + [0] * (history_width - 4) for _ in user_ids],
                pa.int32(),
            ),
            "order_hours": _fixed_list_array(
                [[9, 10, 11, 12] + [0] * (history_width - 4) for _ in user_ids],
                pa.int32(),
            ),
            "days_since_prior_orders": _fixed_list_array(
                [[-1.0, 5.0, 7.0, 8.0] + [0.0] * (history_width - 4) for _ in user_ids],
                pa.float64(),
            ),
            "order_numbers": _fixed_list_array(
                [[1, 2, 3, 4] + [0] * (history_width - 4) for _ in user_ids],
                pa.int32(),
            ),
            "history_length": pa.array([3] * row_count, type=pa.int32()),
            "product_name_length": pa.array(
                [product_name_width] * row_count,
                type=pa.int32(),
            ),
        }
    )


def _write_training_dataset(path, fragment_user_ids):
    path.mkdir()
    for fragment_index, user_ids in enumerate(fragment_user_ids):
        pq.write_table(
            _training_table(user_ids),
            path / f"part-{fragment_index:05d}.parquet",
        )


def _dataset(path, **overrides):
    options = {
        "read_batch_size": 3,
        "batch_size": 2,
        "drop_last": False,
        "shuffle_fragments": False,
        "shuffle_rows": False,
    }
    options.update(overrides)
    return ProductIterableDataset(path, **options)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("read_batch_size", 0, "read_batch_size must be > 0"),
        ("batch_size", 0, "batch_size must be > 0"),
    ],
)
def test_product_dataset_rejects_non_positive_batch_sizes(
    tmp_path,
    argument,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        ProductIterableDataset(tmp_path / "missing", **{argument: value})


def test_product_dataset_rejects_missing_training_columns(tmp_path):
    dataset_path = tmp_path / "training"
    dataset_path.mkdir()
    pq.write_table(
        pa.table({"user_id": pa.array([1], type=pa.int32())}),
        dataset_path / "part-00000.parquet",
    )

    with pytest.raises(ValueError, match="missing columns"):
        ProductIterableDataset(dataset_path)


def test_product_dataset_rejects_incorrect_sequence_width_when_read(tmp_path):
    dataset_path = tmp_path / "training"
    dataset_path.mkdir()
    table = _training_table([1, 2])
    wrong_width = _fixed_list_array(
        [[1] * 99, [2] * 99],
        pa.int32(),
    )
    column_index = table.schema.get_field_index("order_numbers")
    table = table.set_column(column_index, "order_numbers", wrong_width)
    pq.write_table(table, dataset_path / "part-00000.parquet")

    dataset = _dataset(dataset_path)

    with pytest.raises(ValueError):
        list(dataset)


def test_product_dataset_set_epoch_rejects_negative_epoch(tmp_path):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1]])
    dataset = _dataset(dataset_path)

    with pytest.raises(ValueError, match="epoch must be >= 0"):
        dataset.set_epoch(-1)


def test_product_dataset_converts_parquet_rows_to_model_features(tmp_path):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1, 2]])

    batch = next(iter(_dataset(dataset_path)))

    assert set(batch) == {
        "user_id",
        "product_id",
        "aisle_id",
        "department_id",
        "label",
        "product_name_encoded",
        "position_in_order_history",
        "history_order_size",
        "history_reorder_size",
        "is_ordered_history",
        "order_dow_history",
        "order_hour_history",
        "days_since_prior_order_history",
        "order_number_history",
        "next_is_ordered",
        "history_length",
        "sequence_loss_length",
        "product_name_length",
        "is_none",
    }

    assert batch["user_id"].dtype == torch.int64
    assert batch["label"].dtype == torch.int64
    assert batch["days_since_prior_order_history"].dtype == torch.int64
    assert batch["product_name_encoded"].shape == (2, 30)
    assert batch["order_number_history"].shape == (2, 100)

    assert torch.equal(
        batch["next_is_ordered"][0, :4],
        torch.tensor([0, 1, 0, 0]),
    )
    assert torch.equal(
        batch["order_number_history"][0, :4],
        torch.tensor([2, 3, 4, 0]),
    )
    assert torch.allclose(
        batch["days_since_prior_order_history"][0, :4],
        torch.tensor([5, 7, 8, 0]),
    )
    assert torch.equal(batch["history_length"], torch.tensor([3, 3]))
    assert torch.equal(batch["sequence_loss_length"], torch.tensor([2, 2]))
    assert torch.equal(batch["is_none"], torch.tensor([True, False]))


def test_product_dataset_sets_zero_sequence_loss_length_for_one_step_history(
    tmp_path,
):
    dataset_path = tmp_path / "training"
    dataset_path.mkdir()
    table = _training_table([1])
    history_length_index = table.schema.get_field_index("history_length")
    table = table.set_column(
        history_length_index,
        "history_length",
        pa.array([1], type=pa.int32()),
    )
    pq.write_table(table, dataset_path / "part-00000.parquet")

    batch = next(iter(_dataset(dataset_path)))

    assert batch["history_length"].item() == 1
    assert batch["sequence_loss_length"].item() == 0


@pytest.mark.parametrize(
    ("drop_last", "expected_batch_sizes"),
    [
        (True, [2, 2]),
        (False, [2, 2, 1]),
    ],
)
def test_product_dataset_handles_final_incomplete_batch(
    tmp_path,
    drop_last,
    expected_batch_sizes,
):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1, 2, 3, 4, 5]])
    dataset = _dataset(
        dataset_path,
        read_batch_size=3,
        batch_size=2,
        drop_last=drop_last,
    )

    batches = list(dataset)

    assert [len(batch["user_id"]) for batch in batches] == expected_batch_sizes
    assert torch.cat([batch["user_id"] for batch in batches]).tolist() == (
        [1, 2, 3, 4] if drop_last else [1, 2, 3, 4, 5]
    )


def test_product_dataset_carries_incomplete_batch_across_fragments(tmp_path):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1], [2], [3]])

    batches = list(
        _dataset(
            dataset_path,
            read_batch_size=1,
            batch_size=2,
            drop_last=False,
        )
    )

    assert [len(batch["user_id"]) for batch in batches] == [2, 1]
    assert torch.cat([batch["user_id"] for batch in batches]).tolist() == [1, 2, 3]


def test_product_dataset_shuffle_is_repeatable_per_epoch(tmp_path):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1, 2, 3, 4, 5, 6]])
    dataset = _dataset(
        dataset_path,
        read_batch_size=6,
        batch_size=6,
        shuffle_rows=True,
        seed=123,
    )

    epoch_zero_first = next(iter(dataset))["user_id"].tolist()
    epoch_zero_second = next(iter(dataset))["user_id"].tolist()
    dataset.set_epoch(1)
    epoch_one = next(iter(dataset))["user_id"].tolist()

    assert epoch_zero_first == epoch_zero_second
    assert epoch_one != epoch_zero_first
    assert sorted(epoch_one) == [1, 2, 3, 4, 5, 6]


def test_product_dataset_fragment_shuffle_is_repeatable_per_epoch(tmp_path):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1, 2], [3, 4], [5, 6], [7, 8]])
    dataset = _dataset(
        dataset_path,
        read_batch_size=8,
        batch_size=8,
        shuffle_fragments=True,
        seed=123,
    )

    epoch_zero_first = next(iter(dataset))["user_id"].tolist()
    epoch_zero_second = next(iter(dataset))["user_id"].tolist()
    dataset.set_epoch(1)
    epoch_one = next(iter(dataset))["user_id"].tolist()

    assert epoch_zero_first == epoch_zero_second
    assert epoch_one != epoch_zero_first
    assert sorted(epoch_one) == list(range(1, 9))


def test_product_dataset_shards_fragments_without_overlap(tmp_path, mocker):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1, 2], [3, 4], [5, 6], [7, 8]])

    mocked_worker_shard = mocker.patch(
        "instacart_rnn.dataset._worker_shard",
        return_value=(0, 2),
    )
    first_worker_ids = torch.cat(
        [batch["user_id"] for batch in _dataset(dataset_path)]
    ).tolist()

    mocked_worker_shard.return_value = (1, 2)
    second_worker_ids = torch.cat(
        [batch["user_id"] for batch in _dataset(dataset_path)]
    ).tolist()

    assert set(first_worker_ids).isdisjoint(second_worker_ids)
    assert set(first_worker_ids) | set(second_worker_ids) == set(range(1, 9))


def test_worker_shard_combines_dataloader_worker_and_distributed_rank(mocker):
    mocker.patch(
        "instacart_rnn.dataset.get_worker_info",
        return_value=SimpleNamespace(id=2, num_workers=4),
    )
    mocker.patch("instacart_rnn.dataset.dist.is_available", return_value=True)
    mocker.patch("instacart_rnn.dataset.dist.is_initialized", return_value=True)
    mocker.patch("instacart_rnn.dataset.dist.get_rank", return_value=1)
    mocker.patch("instacart_rnn.dataset.dist.get_world_size", return_value=3)

    assert _worker_shard() == (6, 12)


def test_create_product_dataloader_uses_prebatched_dataset(tmp_path):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1, 2, 3]])

    loader = create_product_dataloader(
        dataset_path,
        batch_size=2,
        read_batch_size=3,
        num_workers=0,
        drop_last=False,
        shuffle=False,
    )

    assert loader.batch_size is None
    assert isinstance(loader.dataset, ProductIterableDataset)
    assert [len(batch["user_id"]) for batch in loader] == [2, 1]


def test_product_dataloader_workers_read_each_fragment_once(tmp_path):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1, 2], [3, 4], [5, 6], [7, 8]])
    loader = create_product_dataloader(
        dataset_path,
        batch_size=2,
        read_batch_size=2,
        num_workers=2,
        drop_last=False,
        shuffle=False,
    )

    observed_user_ids = [
        user_id for batch in loader for user_id in batch["user_id"].tolist()
    ]

    assert sorted(observed_user_ids) == list(range(1, 9))
    assert len(observed_user_ids) == len(set(observed_user_ids))


def test_product_dataloader_passes_prefetch_configuration(tmp_path):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1, 2]])

    loader = create_product_dataloader(
        dataset_path,
        num_workers=1,
        prefetch_factor=3,
        shuffle=False,
    )

    assert loader.prefetch_factor == 3


def test_training_column_projection_matches_expected_contract_columns():
    assert PRODUCT_TRAINING_COLUMNS == (
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
