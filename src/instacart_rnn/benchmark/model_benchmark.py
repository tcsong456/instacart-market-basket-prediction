import time

import torch

from instacart_rnn.dataset import create_product_dataloader
from instacart_rnn.models.product_model import ProductModel
from instacart_rnn.training.losses import bce_train_loss

PATH = "gs://instacart-gold-fc45ebb3/training/curated/t2/product_training_data_train"


def train_step(
    *,
    model: ProductModel,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)

    output = model(batch)

    loss = bce_train_loss(output, batch)

    loss.backward()
    optimizer.step()

    return loss


def benchmark_model(
    *,
    model: ProductModel,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    warmup_steps: int = 10,
    benchmark_steps: int = 100,
) -> float:
    """Time in-memory train steps. Excludes DataLoader / Parquet I/O."""
    model.train()

    for _ in range(warmup_steps):
        train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    rows_per_batch = batch["user_id"].size(0)
    start = time.perf_counter()

    for _ in range(benchmark_steps):
        train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    rows_per_second = (rows_per_batch * benchmark_steps) / (time.perf_counter() - start)

    print(
        f"batch_size={rows_per_batch}\n"
        f"iterations={benchmark_steps}\n"
        f"rows/sec={rows_per_second:.2f}\n"
    )

    return rows_per_second


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = create_product_dataloader(
        PATH,
        num_workers=0,
        read_batch_size=4096,
        batch_size=128,
        pin_memory=True,
    )

    batch = {
        key: value.to(device, non_blocking=True)
        for key, value in next(iter(loader)).items()
    }

    print("Loaded batch:", batch["user_id"].size(0), "rows")

    model = ProductModel(
        lstm_size=300,
        dilations=[2**i for i in range(6)],
        filter_widths=[2] * 6,
        skip_channels=64,
        residual_channels=128,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    benchmark_model(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
        warmup_steps=10,
        benchmark_steps=100,
    )


if __name__ == "__main__":
    main()
