import argparse
import logging

from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_platform.pointers import rollback_champion

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore the previous production champion for one mode.",
    )
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--mode", required=True)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    restored = rollback_champion(runs_root=args.runs_root, mode=args.mode)
    logger.info(
        "Restored champion %s for %s; previous stack is %s",
        restored.stack_id,
        restored.mode,
        restored.previous_stack_id,
    )
    print(restored.stack_id, flush=True)


if __name__ == "__main__":
    main()
