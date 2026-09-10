import argparse

from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_rnn.training.runner import (
    InferenceRunConfig,
    run_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model inference and export representations."
    )

    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--eval-path",
        required=True,
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        required=True,
    )
    parser.add_argument(
        "--rows-per-write",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--amp",
        action="store_true",
    )
    parser.add_argument("--pin-memory", action="store_true")

    return parser.parse_args()


def main() -> None:
    configure_logging()

    args = parse_args()

    config = InferenceRunConfig(
        model_name=args.model,
        eval_path=args.eval_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        rows_per_write=args.rows_per_write,
        batch_size=args.batch_size,
        read_batch_size=args.read_batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
        pin_memory=args.pin_memory,
    )

    run_inference(config)


if __name__ == "__main__":
    main()
