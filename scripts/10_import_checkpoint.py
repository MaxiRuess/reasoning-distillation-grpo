"""Explicitly reconstruct historical artifacts before v2 evaluation. Loads model weights on CPU."""

import argparse
import json

from src.data import load_config
from src.recovery import import_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--condition", required=True, choices=["baseline", "sft_traces", "grpo_only", "sft_then_grpo", "re_distill"]
    )
    parser.add_argument("--sft-parent")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(
        json.dumps(
            import_checkpoint(args.source, args.output, load_config(args.config), args.condition, args.sft_parent),
            indent=2,
        )
    )
