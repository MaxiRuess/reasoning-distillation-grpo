import importlib.util
import json
from pathlib import Path

from src.distillation_experiment import analyze_distillation


def test_report_pairs_length_check_on_same_questions_and_keeps_seed_labels(tmp_path, monkeypatch):
    script = Path("scripts/19_generate_distillation_results.py").resolve()
    spec = importlib.util.spec_from_file_location("distillation_report", script)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    monkeypatch.chdir(tmp_path)
    Path("paper").mkdir()
    Path("assets").mkdir()
    proposal = Path("proposal")
    proposal.mkdir()
    models = ["base", *[f"{arm}_{seed}" for arm in ("answer", "trace") for seed in (42, 43, 44)]]
    campaigns = {}
    for tokens, count in ((2048, 256), (4096, 64)):
        campaign = {}
        for model in models:
            rows = []
            for i in range(count):
                if model == "base":
                    correct = i >= 64 if tokens == 2048 else True
                elif model.startswith("answer"):
                    correct = False
                else:
                    correct = {"trace_42": True, "trace_43": i >= 64, "trace_44": i < 128}[model]
                rows.append(
                    {
                        "problem_id": str(i),
                        "ground_truth": "1",
                        "response": r"\boxed{1}" if correct else r"\boxed{0}",
                        "correct": correct,
                        "scorer_version": "boxed-final-v1",
                        "finish_reason": "eos",
                        "generated_tokens": 5,
                    }
                )
            campaign[model] = rows
            path = proposal / "collected" / str(tokens) / model
            path.mkdir(parents=True)
            (path / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        campaigns[tokens] = campaign
    protocol = {
        "protocol_id": "synthetic",
        "subsets": {
            str(tokens): {"rows": [{"problem_id": str(i)} for i in range(count)]}
            for tokens, count in ((2048, 256), (4096, 64))
        },
    }
    (proposal / "protocol.json").write_text(json.dumps(protocol))
    (proposal / "collected/analysis.json").write_text(
        json.dumps(
            {
                "protocol_id": "synthetic",
                "status": "complete",
                "primary_2048": analyze_distillation(campaigns[2048]),
                "secondary_4096": analyze_distillation(campaigns[4096]),
            }
        )
    )
    report.generate(proposal)
    secondary = json.loads((proposal / "collected/descriptive-secondary-analysis.json").read_text())
    # On the same 64 questions, baseline improves from 0% to 100%. Using the full 256 would incorrectly give 25pp.
    assert secondary["short_64_accuracies"]["base"] == 0
    assert secondary["length_comparisons_same_64_questions"]["base"]["delta"] == 1
    text = Path("paper/distillation_results.tex").read_text()
    assert "Teacher trace & 42 & 256" in text
    assert "Teacher trace & 43 & 192" in text
    assert "Teacher trace & 44 & 128" in text
    assert Path("assets/distillation-2026-10-20.png").stat().st_size > 0
