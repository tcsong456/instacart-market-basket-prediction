import argparse

from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_rnn.training.runner import (
    TrainingRunConfig,
    run_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument(
        "--train-path",
        required=True,
    )
    parser.add_argument(
        "--validation-path",
        required=True,
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
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
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument("--early-stopping", type=int, default=3)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
    )
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument(
        "--amp",
        action="store_true",
    )
    parser.add_argument("--pin-memory", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_logging()

    config = TrainingRunConfig(
        model_name=args.model,
        train_path=args.train_path,
        validation_path=args.validation_path,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        read_batch_size=args.read_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=args.amp,
        grad_clip_norm=args.grad_clip_norm,
        warm_start=args.warm_start,
        early_stopping=args.early_stopping,
        pin_memory=args.pin_memory,
    )

    run_training(config)


if __name__ == "__main__":
    main()
