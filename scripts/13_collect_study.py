"""Download completed study predictions for analysis; never run compute."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.artifacts import atomic_json, file_hash, fingerprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output-root", default="outputs/collected")
    args = parser.parse_args()
    ledger = json.loads(Path(args.ledger).read_text())
    campaign = {}
    nodes = ledger["identity"]["plan"]["nodes"]
    if any(
        key not in ledger["results"] or ledger["results"][key]["status"] != "complete"
        for key in [node["id"] for node in nodes]
    ):
        raise ValueError("Only a complete study can generate publication-ready comparisons")
    for node in nodes:
        if node["kind"] != "evaluate":
            continue
        result = ledger["results"][node["id"]]["output"]
        summary = result["benchmarks"][node["benchmark"]]
        remote = summary["results_path"].removeprefix("/vol/")
        destination = Path(args.output_root) / fingerprint(node)[:24]
        destination.mkdir(parents=True, exist_ok=False)
        for filename in ["evaluation.json", "predictions.jsonl", "summary.json", "performance.json"]:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modal",
                    "volume",
                    "get",
                    "reasoning-distillation-vol",
                    remote + "/" + filename,
                    str(destination / filename),
                ],
                check=True,
            )
        verification = ledger["results"][node["id"] + ":verify"]["output"]
        if file_hash(destination / "predictions.jsonl") != verification["predictions_sha256"]:
            raise ValueError("Downloaded results differ from the separately verified volume records")
        if node["benchmark"] == "gsm8k" and node["max_new_tokens"] == 2048 and node["num_samples"] == 1:
            campaign.setdefault(node["condition"], {})[str(node["seed"])] = str(destination / "predictions.jsonl")
    atomic_json(Path(args.output_root) / "campaign.json", campaign)
