import time

import torch

from instacart_rnn.dataset import create_product_dataloader
from instacart_rnn.models.product_input_encoder import ProductInputEncoder
from instacart_rnn.models.product_rnn import ProductRNN

PATH = "gs://instacart-gold-fc45ebb3/training/curated/t2/product_training_data_train"


def compute_sequence_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sequence_loss_length: torch.Tensor,
) -> torch.Tensor:
    # logits / targets: [B, T]
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
    )

    time_indices = torch.arange(
        logits.size(1),
        device=logits.device,
    ).unsqueeze(0)

    # [B, T]
    mask = time_indices < sequence_loss_length.unsqueeze(1)

    masked_losses = losses * mask

    return masked_losses.sum() / mask.sum().clamp_min(1)


class ProductModel(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        rnn: torch.nn.Module,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.rnn = rnn

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ):
        x = self.encoder(batch)

        return self.rnn(
            x,
            batch["history_length"],
        )


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value in batch.items()
    }


def train_step(
    *,
    model: ProductModel,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)

    output = model(batch)

    loss = compute_sequence_loss(
        logits=output.logits,
        targets=batch["next_is_ordered"],
        sequence_loss_length=batch["sequence_loss_length"],
    )

    loss.backward()

    optimizer.step()

    return loss


def benchmark_model_only(
    *,
    model: ProductModel,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    warmup_steps: int = 20,
    benchmark_steps: int = 200,
) -> float:
    model.train()

    # Warm up kernels / CUDA / allocator.
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

    elapsed = time.perf_counter() - start

    total_rows = rows_per_batch * benchmark_steps

    rows_per_second = total_rows / elapsed

    print(f"rows/sec={rows_per_second:.2f}")

    return rows_per_second


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = create_product_dataloader(
        PATH, num_workers=0, read_batch_size=4096, batch_size=128
    )

    batch = next(iter(loader))

    batch = move_batch_to_device(
        batch,
        device,
    )

    encoder = ProductInputEncoder(
        lstm_size=300,
    )

    rnn = ProductRNN(
        input_size=encoder.output_dim,
        lstm_size=300,
        dilations=[2**i for i in range(6)],
        filter_widths=[2] * 6,
        skip_channels=64,
        residual_channels=128,
    )

    model = ProductModel(
        encoder=encoder,
        rnn=rnn,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    benchmark_model_only(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
        warmup_steps=10,
        benchmark_steps=100,
    )
