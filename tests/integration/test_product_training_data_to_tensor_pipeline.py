from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pytest
import torch

from instacart_etl_rnn.common.io import write_parquet
from instacart_etl_rnn.jobs.create_product_training_data_job import (
    run_product_training_data_job,
)
from instacart_rnn.dataset import (
    create_product_dataloader,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)

HISTORY_WIDTH = 100
PRODUCT_NAME_WIDTH = 30

ORDER_DOWS = "0 1 2"
ORDER_HOURS = "10 11 12"
DAYS_SINCE_PRIOR_ORDERS = "-1.0 5.0 7.0"
ORDER_NUMBERS = "1 2 3"

SEQUENCE_WIDTHS = {
    "product_name_encoded": PRODUCT_NAME_WIDTH,
    "is_ordered_history": HISTORY_WIDTH,
    "position_in_order_history": HISTORY_WIDTH,
    "history_order_size": HISTORY_WIDTH,
    "history_reorder_size": HISTORY_WIDTH,
    "order_dows": HISTORY_WIDTH,
    "order_hours": HISTORY_WIDTH,
    "days_since_prior_orders": HISTORY_WIDTH,
    "order_numbers": HISTORY_WIDTH,
}

NONE_PRODUCT_ID = 0


@pytest.fixture(scope="module")
def product_training_path(spark, tmp_path_factory):
    """Run the real Spark training-data job once and return its Parquet directory.

    Three product rows share one user history: two real products plus the
    synthetic ``product_id == 0`` row the loader flags as ``is_none``.

    Args:
        spark: Session-scoped Spark session.
        tmp_path_factory: Factory for the module-scoped temporary directory.

    Returns:
        Path to the Parquet directory written by the job.
    """

    tmp_path = tmp_path_factory.mktemp("product_training")
    product_history = spark.createDataFrame(
        [
            (
                1,
                10,
                0,
                1,
                1,
                "Apple Juice",
                "train",
                "1 0",
                "1 0",
                "2 1",
                "0 0",
                ORDER_DOWS,
                ORDER_HOURS,
                DAYS_SINCE_PRIOR_ORDERS,
                ORDER_NUMBERS,
            ),
            (
                1,
                20,
                1,
                2,
                1,
                "Banana",
                "train",
                "0 1",
                "0 1",
                "2 1",
                "0 0",
                ORDER_DOWS,
                ORDER_HOURS,
                DAYS_SINCE_PRIOR_ORDERS,
                ORDER_NUMBERS,
            ),
            (
                1,
                NONE_PRODUCT_ID,
                0,
                0,
                0,
                "",
                "train",
                "1 0",
                "0 0",
                "2 1",
                "0 0",
                ORDER_DOWS,
                ORDER_HOURS,
                DAYS_SINCE_PRIOR_ORDERS,
                ORDER_NUMBERS,
            ),
        ],
        """
        user_id INT,
        product_id INT,
        label INT,
        aisle_id INT,
        department_id INT,
        product_name STRING,
        eval_set STRING,
        is_ordered_history STRING,
        position_in_order_history STRING,
        history_order_size STRING,
        history_reorder_size STRING,
        order_dows STRING,
        order_hours STRING,
        days_since_prior_orders STRING,
        order_numbers STRING
        """,
    )
    products = spark.createDataFrame(
        [
            (10, "Apple Juice", 1, 1),
            (20, "Banana", 2, 1),
        ],
        """
        product_id INT,
        product_name STRING,
        aisle_id INT,
        department_id INT
        """,
    )

    history_path = tmp_path / "history"
    raw_path = tmp_path / "raw"
    training_path = tmp_path / "training"
    write_parquet(history_path / "product_history_data_train", product_history)
    write_parquet(raw_path / "products", products)

    run_product_training_data_job(
        spark=spark,
        raw_path=str(raw_path),
        input_path=str(history_path),
        output_path=str(training_path),
        contract_path=str(CONTRACT_PATH),
        mode="train",
        min_word_freq=1,
        product_name_length=PRODUCT_NAME_WIDTH,
        encode_length=HISTORY_WIDTH,
    )

    return training_path / "product_training_data_train"


@pytest.fixture(scope="module")
def product_training_table(product_training_path):
    """Read the job output back as an Arrow table for producer-side assertions."""

    return ds.dataset(str(product_training_path), format="parquet").to_table()


@pytest.fixture(scope="module")
def product_rows(product_training_path):
    """Loader output flattened to one feature dict per product_id."""

    return _rows_by_product_id(_load_batches(product_training_path))


def _load_batches(output_path, *, num_workers=0):
    """Stream the Parquet output through the product DataLoader.

    Args:
        output_path: Parquet directory written by the training-data job.
        num_workers: DataLoader worker count.

    Returns:
        List of mini-batches, in loader order.
    """

    loader = create_product_dataloader(
        output_path,
        batch_size=2,
        read_batch_size=2,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )
    return list(loader)


def _rows_by_product_id(batches):
    """Flatten mini-batches into one tensor-per-feature row keyed by product_id.

    Args:
        batches: Mini-batches yielded by the loader.

    Returns:
        Mapping of product_id to that row's feature tensors.
    """

    return {
        batch["product_id"][row_index].item(): {
            name: tensor[row_index] for name, tensor in batch.items()
        }
        for batch in batches
        for row_index in range(len(batch["product_id"]))
    }


def _padded(values, width):
    """Right-pad ``values`` with zeros to ``width``, matching the encoded sequences."""

    return list(values) + [0] * (width - len(values))


def test_spark_product_training_parquet_loads_as_torch_batches(product_training_path):
    batches = _load_batches(product_training_path, num_workers=2)

    assert [len(batch["user_id"]) for batch in batches] == [2, 1]
    assert sorted(
        product_id for batch in batches for product_id in batch["product_id"].tolist()
    ) == [NONE_PRODUCT_ID, 10, 20]

    for batch in batches:
        batch_size = len(batch["user_id"])
        assert batch["product_name_encoded"].shape == (batch_size, PRODUCT_NAME_WIDTH)
        assert batch["next_is_ordered"].shape == (batch_size, HISTORY_WIDTH)
        assert batch["order_number_history"].shape == (batch_size, HISTORY_WIDTH)
        assert batch["label"].dtype == torch.int64
        assert batch["product_id"].dtype == torch.int64
        assert batch["order_number_history"].dtype == torch.int64
        assert batch["is_none"].dtype == torch.bool


def test_loader_keeps_spark_written_sequences_unshifted(product_rows):
    rows = product_rows

    assert rows[10]["user_id"].item() == 1
    assert rows[10]["label"].item() == 0
    assert rows[10]["aisle_id"].item() == 1
    assert rows[10]["department_id"].item() == 1
    assert rows[20]["label"].item() == 1
    assert rows[20]["aisle_id"].item() == 2
    assert rows[10]["is_ordered_history"].tolist() == _padded([1, 0], HISTORY_WIDTH)
    assert rows[20]["is_ordered_history"].tolist() == _padded([0, 1], HISTORY_WIDTH)

    for product_id in (10, 20, NONE_PRODUCT_ID):
        row = rows[product_id]
        assert row["history_order_size"].tolist() == _padded([2, 1], HISTORY_WIDTH)
        assert row["history_reorder_size"].tolist() == _padded([], HISTORY_WIDTH)

    assert rows[10]["position_in_order_history"].tolist() == _padded([1], HISTORY_WIDTH)
    assert rows[20]["position_in_order_history"].tolist() == _padded(
        [0, 1], HISTORY_WIDTH
    )
    assert rows[NONE_PRODUCT_ID]["position_in_order_history"].tolist() == _padded(
        [], HISTORY_WIDTH
    )


def test_loader_left_shifts_next_step_sequences(product_rows):
    rows = product_rows

    for product_id in (10, 20, NONE_PRODUCT_ID):
        row = rows[product_id]
        assert row["order_dow_history"].tolist() == _padded([1, 2], HISTORY_WIDTH)
        assert row["order_hour_history"].tolist() == _padded([11, 12], HISTORY_WIDTH)
        assert row["order_number_history"].tolist() == _padded([2, 3], HISTORY_WIDTH)

    assert rows[20]["next_is_ordered"].tolist() == _padded([1], HISTORY_WIDTH)
    assert rows[10]["next_is_ordered"].tolist() == _padded([], HISTORY_WIDTH)
    assert rows[NONE_PRODUCT_ID]["next_is_ordered"].tolist() == _padded(
        [], HISTORY_WIDTH
    )


def test_loader_preserves_history_length_and_derives_sequence_loss_length(product_rows):
    for row in product_rows.values():
        assert row["history_length"].item() == 2
        assert row["sequence_loss_length"].item() == 1


def test_loader_encodes_product_names_and_marks_none_candidate(product_rows):
    rows = product_rows

    # Vocabulary ids are ranked by word count then alphabetically:
    # apple -> 1, banana -> 2, juice -> 3.
    assert rows[10]["product_name_encoded"].tolist() == _padded(
        [1, 3], PRODUCT_NAME_WIDTH
    )
    assert rows[20]["product_name_encoded"].tolist() == _padded([2], PRODUCT_NAME_WIDTH)
    assert rows[NONE_PRODUCT_ID]["product_name_encoded"].tolist() == _padded(
        [], PRODUCT_NAME_WIDTH
    )

    assert rows[10]["product_name_length"].item() == 2
    assert rows[20]["product_name_length"].item() == 1
    assert rows[NONE_PRODUCT_ID]["product_name_length"].item() == 0

    assert rows[NONE_PRODUCT_ID]["is_none"].item() is True
    assert rows[10]["is_none"].item() is False
    assert rows[20]["is_none"].item() is False


def test_spark_writes_uniform_width_sequences_for_fixed_reshape(product_training_table):
    for column, width in SEQUENCE_WIDTHS.items():
        field_type = product_training_table.schema.field(column).type
        assert pa.types.is_list(field_type)
        assert {
            len(values) for values in product_training_table.column(column).to_pylist()
        } == {width}

    gaps_type = product_training_table.schema.field("days_since_prior_orders").type
    assert gaps_type.value_type == pa.float64()
