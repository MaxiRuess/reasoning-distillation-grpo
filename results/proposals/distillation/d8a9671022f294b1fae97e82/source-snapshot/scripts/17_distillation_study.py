"""Freeze, run and collect the matched-question distillation study within $17.65 remaining."""

import argparse
import json
from pathlib import Path

from src.artifacts import atomic_json, file_hash, fingerprint
from src.data import load_config
from src.distillation_experiment import (
    CAP,
    CONDITIONS,
    build_plan,
    build_quote,
    collect_study,
    execute_campaign,
    freeze_protocol,
    immutable_json,
    study_config,
    validate_proposal,
)


def prepare_proposal():
    from src.pipeline import prepare_phase

    config = study_config(load_config())
    prepared = {}
    for condition in CONDITIONS:
        result = prepare_phase(config, condition, "sft", "data/distillation-prepared")
        manifest, tokens = result["manifest"], result["tokenized_manifest"]
        if manifest["rows"] != 500:
            raise ValueError("Both conditions must retain all 500 shared questions")
        prepared[condition] = {
            "content_hash": manifest["content_hash"],
            "source_ids": manifest["source_ids"],
            "tokens": {key: tokens[key] for key in ("rows", "processed_tokens", "supervised_tokens")},
        }
    if prepared[CONDITIONS[0]]["source_ids"] != prepared[CONDITIONS[1]]["source_ids"]:
        raise ValueError("Training question identities differ")
    protocol = freeze_protocol(config, prepared)
    plan = build_plan(protocol)
    quote = build_quote(plan)
    root = Path("results/proposals/distillation") / protocol["protocol_id"][:24]
    for name, value in (("protocol.json", protocol), ("plan.json", plan), ("quote.json", quote)):
        immutable_json(root / name, value)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=0)
    args = parser.parse_args()
    if not args.execute and not args.collect_only:
        print(json.dumps({"proposal": str(prepare_proposal()), "status": "frozen_no_gpu_jobs"}))
        return
    if not args.proposal or args.execute == args.collect_only:
        parser.error("Supply --proposal and choose exactly one of --execute/--collect-only")
    proposal = Path(args.proposal)
    protocol, plan, quote = validate_proposal(proposal, load_config(), args.budget_usd)
    ledger_path = Path("outputs/distillation") / fingerprint(plan)[:24] / "ledger.json"
    if args.execute:
        immutable_json(
            proposal / "authorization.json",
            {
                "authority": "User: Let's implement this within the budget constraint",
                "scope": "Matched answer versus reasoning-trace SFT, three seeds, fixed evaluation subsets",
                "cap_usd": CAP,
                "envelope_usd": 20,
                "previous_metered_usd": 1.8600004,
                "protocol_id": protocol["protocol_id"],
                "quote_sha256": file_hash(proposal / "quote.json"),
            },
        )
        from modal_train import (
            app,
            prepare_distillation_data,
            prepare_runtime,
            run_evaluation,
            train_distillation,
            verify_volume_result,
        )

        def limited(function, **fixed):
            def call(_timeout, **kwargs):
                return function.with_options(
                    timeout=_timeout, scaledown_window=2, max_containers=4, retries=0, cpu=(2, 2), memory=(16384, 16384)
                ).remote(**fixed, **kwargs)

            return call

        def no_grpo(**kwargs):
            raise ValueError("GRPO is outside this distillation study")

        state = {"status": "starting", "proposal": str(proposal), "ledger": str(ledger_path), "cap_usd": CAP}
        try:
            with app.run():
                state.update(status="running", app_id=app.app_id)
                atomic_json(Path("outputs/distillation-active.json"), state)
                ledger = execute_campaign(
                    plan,
                    protocol["config"],
                    quote,
                    CAP - quote["non_runtime_allowance_usd"],
                    prepare=limited(prepare_distillation_data, protocol=protocol),
                    sft=limited(train_distillation, protocol=protocol),
                    grpo=no_grpo,
                    evaluate=limited(run_evaluation),
                    verify=limited(verify_volume_result),
                    prepare_runtime=limited(prepare_runtime),
                    root="outputs/distillation",
                )
            state["status"] = "jobs_complete_app_closed"
        finally:
            if state["status"] != "jobs_complete_app_closed":
                state["status"] = "failed_app_context_closed_reconcile_before_retry"
            atomic_json(Path("outputs/distillation-active.json"), state)
    else:
        ledger = json.loads(ledger_path.read_text())
    result = collect_study(proposal, ledger)
    print(json.dumps({"ledger": str(ledger_path), "analysis": result}, indent=2))


if __name__ == "__main__":
    main()
