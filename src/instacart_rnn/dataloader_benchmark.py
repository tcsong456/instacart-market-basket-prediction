import statistics
import time

from instacart_rnn.dataset import create_product_dataloader

PATH = "gs://instacart-gold-fc45ebb3/training/curated/t2/product_training_data_train"


def benchmark_loader(loader, max_batches=500):
    start = time.perf_counter()

    rows = 0
    batches = 0

    for batch in loader:
        rows += len(batch["user_id"])
        batches += 1

        if batches >= max_batches:
            break

    elapsed = time.perf_counter() - start

    return rows / elapsed


def benchmark_config(
    config: dict,
    *,
    repeats: int = 1,
    max_batches: int = 500,
) -> float:
    results = []

    for _ in range(repeats):
        loader = create_product_dataloader(PATH, **config)

        throughput = benchmark_loader(
            loader,
            max_batches=max_batches,
        )

        results.append(throughput)

    return statistics.median(results)


if __name__ == "__main__":
    configs = [
        {"num_workers": 0} | {"read_batch_size": read_batch_size}
        for read_batch_size in [1024, 2048, 4096, 8192, 16384]
    ]

    results = {}

    for config in configs:
        throughput = benchmark_config(
            config,
        )

        key = (config["num_workers"], config["read_batch_size"])
        results[key] = throughput

        print(f"config={config}: {throughput:.2f} rows/sec")

    best_workers = max(
        results,
        key=results.get,
    )

    print(f"best num_workers: {best_workers}")
    print(f"throughput: {results[best_workers]:.2f} rows/sec")
