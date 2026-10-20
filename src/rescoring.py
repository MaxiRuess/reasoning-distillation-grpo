"""Rescore retained generations on CPU without changing their source artifacts."""

import json
from pathlib import Path

from src.artifacts import atomic_json, file_hash, fingerprint, runtime_metadata
from src.evaluation import summarize_records
from src.reward import SCORER_VERSION, score_response


def rescore_saved_evaluation(source, output_root, scorer=SCORER_VERSION):
    source = Path(source).resolve()
    raw = (source / "predictions.jsonl").read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Only complete prediction transactions can be rescored")
    rows = [json.loads(line) for line in raw.splitlines()]
    manifest = json.loads((source / "evaluation.json").read_text())
    original = json.loads((source / "summary.json").read_text())
    if scorer == SCORER_VERSION and manifest.get("identity", {}).get("answer_contract") != SCORER_VERSION:
        raise ValueError(
            "Original generations did not use this answer contract; choose their archived scorer explicitly"
        )
    ids, n = manifest["problem_ids"], manifest["num_samples"]
    summarize_records(rows, ids, n)  # Reject duplicates, missing rows and reordered samples.
    identity = {
        "kind": "cpu_rescore",
        "source_run_id": original["run_id"],
        "source_predictions_sha256": file_hash(source / "predictions.jsonl"),
        "source_evaluation_sha256": file_hash(source / "evaluation.json"),
        "source_summary_sha256": file_hash(source / "summary.json"),
        "scorer_version": scorer,
        "implementation": {
            name: file_hash(Path(__file__).with_name(name))
            for name in [
                "reward.py",
                "rescoring.py",
                "scoring_v2.py",
                "scoring_v3.py",
                "scoring_v4.py",
                "scoring_v5.py",
                "scoring_v6.py",
            ]
        },
        "runtime": runtime_metadata(),
    }
    rescored = [
        {**row, **score_response(row["response"], row["ground_truth"], original["benchmark"], scorer).to_dict()}
        for row in rows
    ]
    path = Path(output_root) / fingerprint(identity)[:24]
    summary = {
        **summarize_records(rescored, ids, n),
        "benchmark": original["benchmark"],
        "model_identity": original["model_identity"],
        "scorer_version": scorer,
        "run_id": fingerprint(identity),
        "results_path": str(path),
        "pilot_or_subset": original.get("pilot_or_subset", False),
        "generation_repeated": False,
        "training_repeated": False,
        "source_run_id": original["run_id"],
    }
    path.mkdir(parents=True, exist_ok=False)
    (path / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rescored)
    )
    atomic_json(path / "rescore.json", identity)
    atomic_json(path / "summary.json", summary)
    return summary
