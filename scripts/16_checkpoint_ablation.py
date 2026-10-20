"""Freeze or execute the $20, three-construction historical checkpoint ablation.

Default preparation is local CPU work only. Execution requires the exact frozen
proposal and an explicit cap; no training jobs or automatic paid retries exist.
"""

import argparse
import json
from pathlib import Path

from src.artifacts import file_hash, fingerprint
from src.checkpoint_experiment import (
    collect_experiment,
    immutable_json,
    prepare_proposal,
    validate_proposal,
)
from src.data import load_config
from src.study import execute_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", help="Existing content-addressed proposal directory")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=0)
    args = parser.parse_args()
    config = load_config()
    if not args.execute and not args.collect_only:
        print(json.dumps({"proposal": str(prepare_proposal(config)), "status": "frozen_not_executed"}))
        return
    if not args.proposal or (args.execute and args.collect_only):
        parser.error("Choose execution or collection and supply --proposal")
    proposal = Path(args.proposal)
    protocol, plan, _quote = validate_proposal(proposal, config, args.budget_usd)
    ledger_path = Path("outputs/checkpoint-ablation") / fingerprint(plan)[:24] / "ledger.json"
    if args.execute:
        immutable_json(
            proposal / "authorization.json",
            {
                "scope": "Implementation of the accepted 256-question, three-model evaluation-only experiment",
                "budget_usd": args.budget_usd,
                "protocol_id": protocol["protocol_id"],
                "quote_sha256": file_hash(proposal / "quote.json"),
                "authority": "User requested implementation of the preceding experiment design within its $20 budget",
            },
        )
        from modal_train import app, prepare_runtime, reconstruct_checkpoint, run_evaluation, verify_volume_result

        def limited(function):
            def call(_timeout, **kwargs):
                return function.with_options(
                    timeout=_timeout,
                    scaledown_window=2,
                    max_containers=1,
                    retries=0,
                    cpu=(2, 2),
                    memory=(16384, 16384),
                ).remote(**kwargs)

            return call

        def reconstruct(**kwargs):
            return limited(reconstruct_checkpoint)(protocol=protocol, **kwargs)

        def no_training(**kwargs):
            raise ValueError("Training and data regeneration are outside this experiment")

        with app.run():
            ledger = execute_study(
                plan,
                config,
                str(proposal / "quote.json"),
                args.budget_usd,
                prepare=no_training,
                sft=no_training,
                grpo=no_training,
                evaluate=limited(run_evaluation),
                verify=limited(verify_volume_result),
                prepare_runtime=limited(prepare_runtime),
                reconstruct=reconstruct,
                ledger_root="outputs/checkpoint-ablation",
            )
    else:
        ledger = json.loads(ledger_path.read_text())
    result = collect_experiment(proposal, ledger, proposal / "collected")
    print(json.dumps({"ledger": str(ledger_path), "analysis": result}, indent=2))


if __name__ == "__main__":
    main()
