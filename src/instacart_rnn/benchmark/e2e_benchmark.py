import time

import torch
import torch.nn.functional as F

from instacart_rnn.dataset import create_product_dataloader
from instacart_rnn.models.product_input_encoder import ProductInputEncoder
from instacart_rnn.models.product_rnn import ProductRNN

PATH = "gs://instacart-gold-fc45ebb3/training/curated/t2/product_training_data_train"


def compute_sequence_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sequence_loss_length: torch.Tensor,
) -> torch.Tensor:
    """
    logits:
        [B, T]

    targets:
        [B, T]

    sequence_loss_length:
        [B]

    Only timesteps:
        0 <= t < sequence_loss_length[row]
    contribute to the loss.
    """

    # [B, T]
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
    )

    # [1, T]
    time_indices = torch.arange(
        logits.size(1),
        device=logits.device,
    ).unsqueeze(0)

    # [B, T]
    mask = time_indices < sequence_loss_length.unsqueeze(1)

    masked_losses = losses * mask

    valid_positions = mask.sum()

    if valid_positions == 0:
        return logits.sum() * 0.0

    return masked_losses.sum() / valid_positions


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


def train_step(
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


def benchmark_model(
    model: ProductModel,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    iterations: int = 100,
    warmup: int = 10,
) -> float:
    """
    Benchmark model training throughput using the same
    in-memory batch repeatedly.

    This deliberately excludes DataLoader / Parquet I/O.
    """

    model.train()

    batch = {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value in batch.items()
    }

    #
    # Warmup
    #
    # This avoids timing:
    # - initial CUDA setup
    # - kernel initialization
    # - allocator startup
    # - other first-step overhead
    #
    for _ in range(warmup):
        train_step(
            model,
            batch,
            optimizer,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iterations):
        train_step(
            model,
            batch,
            optimizer,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    batch_size = len(batch["user_id"])
    rows = batch_size * iterations

    rows_per_second = rows / elapsed

    print(
        f"batch_size={batch_size}\n"
        f"iterations={iterations}\n"
        f"rows/sec={rows_per_second:.2f}\n"
    )

    return rows_per_second


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = create_product_dataloader(
        PATH, num_workers=0, read_batch_size=4096, batch_size=128
    )

    batch = next(iter(loader))

    print(
        "Loaded batch:",
        len(batch["user_id"]),
        "rows",
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

    benchmark_model(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
        iterations=100,
        warmup=10,
    )


if __name__ == "__main__":
    main()
