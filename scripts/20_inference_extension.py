"""Freeze, run or collect the separately authorized $30 inference-only follow-up."""

import argparse
import json
from pathlib import Path

from src.artifacts import atomic_json, file_hash, fingerprint
from src.checkpoint_experiment import immutable_json
from src.inference_extension import CAP, build_plan, build_quote, collect, execute, freeze_protocol, validate_proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=0)
    args = parser.parse_args()
    if not args.execute and not args.collect_only:
        protocol = freeze_protocol()
        proposal = Path("results/proposals/inference-extension") / protocol["protocol_id"][:24]
        plan = build_plan(protocol)
        for name, value in (("protocol", protocol), ("plan", plan), ("quote", build_quote(plan))):
            immutable_json(proposal / (name + ".json"), value)
        source = proposal / "source-snapshot"
        for file, digest in protocol["implementation"].items():
            target = source / file
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and file_hash(target) != digest:
                raise ValueError("Frozen source snapshot changed")
            target.write_bytes(Path(file).read_bytes())
        print(
            json.dumps(
                {
                    "proposal": str(proposal),
                    "status": "frozen_no_cloud_jobs",
                    "reserved_usd": build_quote(plan)["reserved_runtime_usd"] + 1,
                }
            )
        )
        return
    if not args.proposal or args.execute == args.collect_only:
        parser.error("Choose exactly one of --execute/--collect-only and supply --proposal")
    proposal = Path(args.proposal)
    protocol, plan, _quote = validate_proposal(proposal, args.budget_usd)
    ledger_path = Path("outputs/inference-extension") / fingerprint(plan)[:24] / "ledger.json"
    state_path = Path("outputs/inference-extension-active.json")
    if args.execute:
        immutable_json(
            proposal / "authorization.json",
            {
                "authority": "User: Let's implement this please, accepting the 128-question inference package and new $30 cap",
                "cap_usd": CAP,
                "protocol_id": protocol["protocol_id"],
                "quote_sha256": file_hash(proposal / "quote.json"),
                "scope": "Existing student and three trace seeds; sampled 2K/4K/8K; no-thinking sampled and original-greedy diagnostics; no training",
            },
        )
        from modal_train import app, evaluate_inference_extension, prepare_inference_extension, verify_volume_result

        def limited(function):
            def call(_timeout, **kwargs):
                return function.with_options(
                    timeout=_timeout, scaledown_window=2, max_containers=4, retries=0, cpu=(2, 2), memory=(16384, 16384)
                ).remote(**kwargs)

            return call

        def evaluate(_timeout, condition, **kwargs):
            job = protocol["jobs"][condition]
            if (
                kwargs["model_path"] != job["model_path"]
                or kwargs["max_new_tokens"] != job["generation"]["max_new_tokens"]
            ):
                raise ValueError("Executor requested an unfrozen checkpoint or output limit")
            return limited(evaluate_inference_extension)(_timeout=_timeout, protocol=protocol, job_id=condition)

        state = {"status": "starting", "proposal": str(proposal), "ledger": str(ledger_path), "cap_usd": CAP}
        try:
            with app.run():
                state.update(status="running", app_id=app.app_id)
                atomic_json(state_path, state)
                ledger = execute(
                    protocol,
                    preflight=lambda _timeout: limited(prepare_inference_extension)(
                        _timeout=_timeout, protocol=protocol
                    ),
                    evaluate=evaluate,
                    verify=limited(verify_volume_result),
                )
            state["status"] = "jobs_complete_app_closed"
        finally:
            if state["status"] != "jobs_complete_app_closed":
                state["status"] = "failed_app_context_closed_reconcile_before_retry"
            atomic_json(state_path, state)
    else:
        ledger = json.loads(ledger_path.read_text())
    result = collect(proposal, ledger)
    state = json.loads(state_path.read_text())
    state.update(status="complete_app_closed", analysis=str(proposal / "collected/analysis.json"))
    atomic_json(state_path, state)
    print(json.dumps({"ledger": str(ledger_path), "primary_8192": result["primary_8192"]}, indent=2))


if __name__ == "__main__":
    main()
