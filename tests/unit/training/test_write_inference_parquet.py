from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

from instacart_rnn.training.export import write_inference_parquet

DEVICE = torch.device("cpu")
BATCH_TENSOR_NAMES = ("user_id", "product_id", "aisle_id")
OUTPUT_TENSOR_NAMES = ("final_states", "final_logits")
STATE_WIDTH = 4
REPRESENTATION_FILENAME = "tinymodel_representation.parquet"


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, batch):
        pred = self.linear(batch["x"])
        final_logits = pred.squeeze(-1)

        return SimpleNamespace(
            logits=final_logits.unsqueeze(-1).expand(-1, 2).contiguous(),
            final_logits=final_logits,
            final_states=final_logits.unsqueeze(-1)
            .expand(
                -1,
                STATE_WIDTH,
            )
            .contiguous(),
        )


def _batch(*, x, user_ids):
    row_count = x.size(0)

    return {
        "user_id": user_ids,
        "product_id": torch.arange(1, row_count + 1, dtype=torch.long),
        "aisle_id": torch.full((row_count,), 7, dtype=torch.long),
        "x": x,
    }


def _write_inference_parquet(*, output_path, dataloader, rows_per_write):
    model = TinyModel()
    model.linear.weight.data.fill_(0.5)

    write_inference_parquet(
        model=model,
        dataloader=dataloader,
        device=DEVICE,
        output_path=str(output_path),
        rows_per_write=rows_per_write,
        batch_tensor_names=BATCH_TENSOR_NAMES,
        output_tensor_names=OUTPUT_TENSOR_NAMES,
    )

    return output_path / REPRESENTATION_FILENAME


@pytest.mark.parametrize("rows_per_write", [0, -1])
def test_write_inference_parquet_rejects_non_positive_rows_per_write(
    tmp_path,
    rows_per_write,
):
    batch = _batch(
        x=torch.ones(1, 1),
        user_ids=torch.tensor([1]),
    )

    with pytest.raises(ValueError, match="rows_per_write must be greater than 0"):
        write_inference_parquet(
            model=TinyModel(),
            dataloader=[batch],
            device=DEVICE,
            output_path=str(tmp_path),
            rows_per_write=rows_per_write,
            batch_tensor_names=BATCH_TENSOR_NAMES,
            output_tensor_names=OUTPUT_TENSOR_NAMES,
        )


def test_write_inference_parquet_writes_leftover_rows_and_probabilities(
    tmp_path,
):
    batches = [
        _batch(
            x=torch.tensor([[0.0], [2.0]]),
            user_ids=torch.tensor([11, 22]),
        ),
        _batch(
            x=torch.tensor([[4.0]]),
            user_ids=torch.tensor([33]),
        ),
    ]

    parquet_path = _write_inference_parquet(
        output_path=tmp_path,
        dataloader=batches,
        rows_per_write=2,
    )

    parquet_file = pq.ParquetFile(parquet_path)
    table = parquet_file.read()
    states_type = table.schema.field("final_states").type

    assert parquet_file.num_row_groups == 2
    assert table.num_rows == 3
    assert table.column("user_id").to_pylist() == [11, 22, 33]
    assert table.column("product_id").to_pylist() == [1, 2, 1]
    assert table.column("aisle_id").to_pylist() == [7, 7, 7]
    assert pa.types.is_fixed_size_list(states_type)
    assert states_type.list_size == STATE_WIDTH

    logits = torch.tensor(table.column("final_logits").to_pylist())
    probabilities = table.column("final_probabilities").to_pylist()

    assert logits.tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert probabilities == pytest.approx(torch.sigmoid(logits).tolist())


def test_write_inference_parquet_writes_gcs_url_through_filesystem(
    tmp_path,
    mocker,
):
    local_file = tmp_path / "gcs.parquet"
    fake_fs = mocker.Mock()
    fake_fs.open.side_effect = lambda path, mode: local_file.open(mode)
    mocker.patch(
        "instacart_rnn.training.export.gcsfs.GCSFileSystem",
        return_value=fake_fs,
    )
    batch = _batch(
        x=torch.tensor([[2.0]]),
        user_ids=torch.tensor([11]),
    )

    write_inference_parquet(
        model=TinyModel(),
        dataloader=[batch],
        device=DEVICE,
        output_path="gs://bucket/output",
        rows_per_write=10,
        batch_tensor_names=BATCH_TENSOR_NAMES,
        output_tensor_names=OUTPUT_TENSOR_NAMES,
    )

    fake_fs.open.assert_called_once_with(
        f"gs://bucket/output/{REPRESENTATION_FILENAME}",
        "wb",
    )

    table = pq.read_table(local_file)

    assert table.num_rows == 1
    assert table.column("user_id").to_pylist() == [11]
