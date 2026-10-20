import json
import runpy
import shutil
from pathlib import Path

import pytest


def test_labels_stay_attached_to_comparison_ids_after_json_key_sorting(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "results/recovery").mkdir(parents=True)
    (tmp_path / "paper").mkdir()
    (tmp_path / "assets").mkdir()
    source = root / "results/recovery/historical-paired-analysis.json"
    # The real atomic JSON writer sorts object keys. Rendering may not rely on insertion order.
    shutil.copy(source, tmp_path / "results/recovery/historical-paired-analysis.json")
    captured = []
    from matplotlib.axes import Axes

    original = Axes.errorbar

    def capture(self, x, y, *args, **kwargs):
        captured.extend(x)
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture)
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(root / "scripts/14_generate_paper_results.py"))["main"]()
    pairs = (tmp_path / "paper/historical_pairs.tex").read_text()
    counts = json.loads(source.read_text())["conditions"]
    keys = [
        ("re_distill", "sft_traces", "Open-R1 SFT minus s1K SFT"),
        ("grpo_only", "base_model", "GRPO minus starting checkpoint"),
        ("sft_then_grpo", "sft_traces", r"Combined$^*$ minus s1K SFT"),
        ("sft_traces", "base_model", "s1K SFT minus starting checkpoint"),
        ("re_distill", "grpo_only", "Open-R1 SFT minus GRPO"),
    ]
    expected = [(counts[a]["num_correct"] - counts[b]["num_correct"]) / 1319 * 100 for a, b, _ in keys]
    for (_, _, label), value in zip(keys, expected, strict=True):
        assert label + " & " + f"{value:+.2f}" in pairs
    assert captured == pytest.approx(expected)
