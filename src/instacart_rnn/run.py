import argparse

from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_rnn.training.exec_run import execute_run
from instacart_rnn.training.runner import InferenceRunConfig, TrainingRunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=256,
    )
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--lstm-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--max-candidate",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--rows-per-write",
        type=int,
        default=100000,
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
    parser.add_argument("--early-stopping", type=int, default=2)
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
    )
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--image", default="")
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

    base_path = f"{args.data_root.rstrip('/')}/{args.model}_training_data"

    train_path = f"{base_path}_train"
    validation_path = f"{base_path}_validation"
    eval_path = f"{base_path}_evaluation"

    train_config = TrainingRunConfig(
        model_name=args.model,
        train_path=train_path,
        validation_path=validation_path,
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        read_batch_size=args.read_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lstm_size=args.lstm_size,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=args.amp,
        grad_clip_norm=args.grad_clip_norm,
        warm_start=args.warm_start,
        early_stopping=args.early_stopping,
        pin_memory=args.pin_memory,
    )

    inference_config = InferenceRunConfig(
        model_name=args.model,
        eval_path=eval_path,
        checkpoint_path="",
        output_path="",
        rows_per_write=args.rows_per_write,
        batch_size=args.eval_batch_size,
        read_batch_size=args.read_batch_size,
        lstm_size=args.lstm_size,
        max_candidate=args.max_candidate,
        num_workers=args.num_workers,
        amp=args.amp,
        pin_memory=args.pin_memory,
    )

    execute_run(
        run_id=args.run_id,
        runs_root=args.runs_root,
        training_config=train_config,
        inference_config=inference_config,
        git_commit=args.git_commit,
        image=args.image,
    )


if __name__ == "__main__":
    main()
