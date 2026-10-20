"""Default: print the full study without importing Modal or starting any remote work."""

import argparse
import json

from src.data import load_config
from src.study import build_pilot_plan, build_study_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", choices=["legacy", "controlled", "pilot"], default="controlled")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--prepare-only", action="store_true", help="Prepare the whole campaign locally on CPU")
    parser.add_argument("--quote")
    parser.add_argument("--budget-usd", type=float, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = build_pilot_plan(config) if args.experiment == "pilot" else build_study_plan(config, args.experiment)
    if args.prepare_only:
        from src.pipeline import prepare_phase

        for node in plan["nodes"]:
            if node["kind"] == "prepare":
                result = prepare_phase(config, node["condition"], node["phase"], "data/prepared")
                print(node["id"], result["rows"], result["manifest"]["path"])
        return
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if not args.quote or args.budget_usd <= 0:
        parser.error("Execution requires a measured quote and an explicitly agreed --budget-usd")
    if args.config != "configs/config.yaml":
        parser.error("Modal currently bundles configs/config.yaml; custom config cannot silently diverge")
    from src.study import validate_quote

    with open(args.quote) as stream:
        validate_quote(plan, json.load(stream), args.budget_usd)
    from modal_train import app, run_full_pipeline

    with app.run():
        run_full_pipeline(experiment=args.experiment, dry_run=False, quote=args.quote, budget_usd=args.budget_usd)


if __name__ == "__main__":
    main()
