"""Produce a reviewable initial pilot reservation; do not start remote work."""

import argparse
import json
from pathlib import Path

from src.artifacts import atomic_json, fingerprint
from src.data import load_config
from src.study import build_pilot_plan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/proposals")
    args = parser.parse_args()
    config = load_config()
    plan = build_pilot_plan(config)
    pricing = json.loads(Path("configs/modal-pricing-2026-10-19.json").read_text())
    cpu_rate = pricing["cpu_core_per_second"] * 2 + pricing["memory_gib_per_second"] * 16
    timeouts = [3600, 3600, 300, 600, 900, 600, 60]
    tasks = {
        node["id"]: {
            "rate_per_second": cpu_rate + pricing["gpu_per_second"].get(node["gpu"], 0),
            "timeout_seconds": timeout,
            "overhead_seconds": 120,
        }
        for node, timeout in zip(plan["nodes"], timeouts, strict=True)
    }
    quote = {
        "plan_digest": fingerprint(plan),
        "pricing_source": pricing,
        "tasks": tasks,
        "pilot_evidence": ["initial smoke pilot: measurements not yet available"],
        "status": "proposal_not_approved",
        "suggested_cap_usd": 10,
        "runtime_reservations_usd": sum(
            task["rate_per_second"] * (task["timeout_seconds"] + 120) for task in tasks.values()
        ),
    }
    directory = Path(args.output_root) / ("pilot-" + fingerprint(quote)[:24])
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in [("quote.json", quote), ("plan.json", plan)]:
        path = directory / name
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise FileExistsError(f"Refusing to overwrite changed proposal artifact: {path}")
        else:
            atomic_json(path, value)
    print(
        json.dumps(
            {"quote_path": str(directory / "quote.json"), "plan_path": str(directory / "plan.json"), "proposal": quote},
            indent=2,
        )
    )
