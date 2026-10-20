"""Generate paper tables/figure and descriptive paired checks from the completed frozen study. CPU only."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.artifacts import atomic_json, file_hash
from src.checkpoint_experiment import immutable_json
from src.distillation_experiment import SEEDS, numeric_final_score
from src.statistics import paired_comparison


def mean_item_contrast(left, right):
    for a, b in zip(left, right, strict=True):
        paired_comparison(a, b, replicates=1)
    deltas = np.array(
        [
            [int(a["correct"]) - int(b["correct"]) for a, b in zip(aa, bb, strict=True)]
            for aa, bb in zip(left, right, strict=True)
        ]
    )
    items = deltas.mean(axis=0)
    rng = np.random.default_rng(42)
    draws = np.concatenate([items[rng.integers(len(items), size=(500, len(items)))].mean(axis=1) for _ in range(20)])
    return {
        "mean_delta": float(items.mean()),
        "paired_item_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "per_seed_delta": deltas.mean(axis=1).tolist(),
        "training_seed_delta_std": float(deltas.mean(axis=1).std(ddof=1)) if len(left) > 1 else None,
        "num_distinct_questions": len(items),
        "descriptive_secondary": True,
    }


def generate(proposal):
    proposal = Path(proposal)
    analysis = json.loads((proposal / "collected/analysis.json").read_text())
    protocol = json.loads((proposal / "protocol.json").read_text())
    if analysis["status"] != "complete" or analysis["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("A complete source-bound study is required")
    models = ["base", *[f"{arm}_{seed}" for seed in SEEDS for arm in ("answer", "trace")]]
    campaigns = {}
    for tokens, count in ((2048, 256), (4096, 64)):
        campaigns[tokens] = {}
        for model in models:
            path = proposal / "collected" / str(tokens) / model / "predictions.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != count or [r["problem_id"] for r in rows] != [
                r["problem_id"] for r in protocol["subsets"][str(tokens)]["rows"]
            ]:
                raise ValueError("Incomplete or mismatched test questions")
            campaigns[tokens][model] = [{**r, **numeric_final_score(r["response"], r["ground_truth"])} for r in rows]
    primary = analysis["primary_2048"]["primary"]
    main = campaigns[2048]
    check = mean_item_contrast([main[f"trace_{s}"] for s in SEEDS], [main[f"answer_{s}"] for s in SEEDS])
    if not np.isclose(check["mean_delta"], primary["mean_delta"]):
        raise ValueError("Paper effect does not reconcile with the frozen analysis")
    baseline_checks = {
        arm: mean_item_contrast([main[f"{arm}_{s}"] for s in SEEDS], [main["base"]] * 3) for arm in ("answer", "trace")
    }
    long = campaigns[4096]
    ids = {r["problem_id"] for r in long["base"]}
    short = {model: [r for r in rows if r["problem_id"] in ids] for model, rows in main.items()}
    length_checks = {model: paired_comparison(long[model], short[model]) for model in models}
    mean_length_checks = {
        arm: mean_item_contrast([long[f"{arm}_{s}"] for s in SEEDS], [short[f"{arm}_{s}"] for s in SEEDS])
        for arm in ("answer", "trace")
    }
    baseline_long_checks = {
        arm: mean_item_contrast([long[f"{arm}_{s}"] for s in SEEDS], [long["base"]] * 3) for arm in ("answer", "trace")
    }
    immutable_json(proposal / "collected/parent-comparisons-4096.json", baseline_long_checks)
    secondary = {
        "protocol_id": protocol["protocol_id"],
        "source_analysis_sha256": file_hash(proposal / "collected/analysis.json"),
        "base_comparisons_2048": baseline_checks,
        "length_comparisons_same_64_questions": length_checks,
        "mean_length_comparisons_same_64_questions": mean_length_checks,
        "short_64_accuracies": {m: sum(r["correct"] for r in rows) / 64 for m, rows in short.items()},
        "caution": "Descriptive secondary analyses; item intervals conditional on observed training seeds; equal output ceilings are not equal consumed compute",
    }
    immutable_json(proposal / "collected/descriptive-secondary-analysis.json", secondary)

    def acc(rows):
        return 100 * sum(r["correct"] for r in rows) / len(rows)

    mean_answer = np.mean([acc(main[f"answer_{s}"]) for s in SEEDS])
    mean_trace = np.mean([acc(main[f"trace_{s}"]) for s in SEEDS])
    low, high = [100 * v for v in primary["paired_item_bootstrap_95"]]
    delta, seed_sd = 100 * primary["mean_delta"], 100 * primary["training_seed_delta_std"]
    base_delta = baseline_checks["trace"]
    bl, bh = [100 * v for v in base_delta["paired_item_bootstrap_95"]]
    abstract = (
        f"At 2K, trace SFT averaged {mean_trace:.2f}\\% accuracy versus {acc(main['base']):.2f}\\% "
        f"for the unchanged student, a descriptive mean difference of {100 * base_delta['mean_delta']:+.2f} "
        f"percentage points (95\\% paired item-bootstrap interval {bl:+.2f} to {bh:+.2f}, conditional on the trained seeds). "
        f"On the preselected 64-question 4K check, trace SFT averaged "
        f"{np.mean([acc(long[f'trace_{s}']) for s in SEEDS]):.2f}\\% versus {acc(long['base']):.2f}\\% "
        f"unchanged; this check did not demonstrate an advantage. Answer-only SFT averaged {mean_answer:.2f}\\% at 2K.\n"
    )
    Path("paper/distillation_abstract_results.tex").write_text(abstract)
    lines = [
        r"\section{Results}",
        r"\subsection{Accuracy at the 2K output limit}",
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Seed & Correct/256 & Numeric (\%) & Strict (\%) & Unfinished (\%) & Tokens \\",
        r"\midrule",
    ]
    for model in models:
        arm = "Unchanged" if model == "base" else ("Answer only" if model.startswith("answer") else "Teacher trace")
        seed = "--" if model == "base" else model.split("_")[1]
        stats = analysis["primary_2048"]["arms"][model]
        lines.append(
            f"{arm} & {seed} & {sum(r['correct'] for r in main[model])} & {acc(main[model]):.2f} & "
            f"{100 * stats['strict_accuracy']:.2f} & {100 * stats['unfinished_thinking_rate']:.2f} & {stats['mean_generated_tokens']:.0f} "
            + r"\\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\caption{All seven checkpoints on the same 256 GSM8K questions at a 2,048-token output limit. Numeric is the predeclared numerical endpoint; strict requires exactly one final box. Unfinished denotes unfinished thinking. Tokens are mean useful generated tokens per response.}",
        r"\label{tab:distill-results}\end{table}",
        (
            f"Trace SFT averaged {mean_trace:.2f}\\% numerical accuracy, compared with {acc(main['base']):.2f}\\% "
            "for the unchanged student (Table~\\ref{tab:distill-results}). All three trace checkpoints exceeded the unchanged "
            f"student on these questions. The descriptive mean difference was {100 * base_delta['mean_delta']:+.2f} "
            f"percentage points (95\\% conditional item interval {bl:+.2f} to {bh:+.2f}); "
            f"the seed-difference standard deviation was {100 * base_delta['training_seed_delta_std']:.2f} points. "
            "This unchanged-student comparison is secondary to the protocol's primary trace-minus-answer comparison."
        ),
        r"\begin{figure}[ht]\centering\includegraphics[width=\textwidth]{assets/distillation-2026-10-20.png}",
        r"\caption{Left: 2K numerical accuracy on all 256 questions, showing each trained seed and the unchanged student. Right: the descriptive length check on the same preselected 64 questions at both limits. Thin lines show individual trace seeds; the thicker line shows their mean. The panels use different question sets and vertical scales. Neither panel shows equal consumed compute.}\label{fig:distill}\end{figure}",
        r"\subsection{Answer-only training produced a collapsed comparator}",
        (
            f"Answer-only SFT averaged {mean_answer:.2f}\\% at 2K. The predeclared primary trace-minus-answer "
            f"difference was {delta:+.2f} percentage points (95\\% conditional item interval {low:+.2f} to {high:+.2f}); "
            f"the seed-difference standard deviation was {seed_sd:.2f} points. "
            "All 768 answer-only responses contained empty visible thinking blocks, generally followed by incorrect numerical "
            "guesses. Their failure was therefore not an unfinished-thinking scoring penalty. This large contrast describes "
            "the two executed recipes, including the answer-only template behavior. It does not isolate a general benefit "
            "of teacher traces over a well-tuned answer-only alternative."
        ),
        r"\subsection{A longer output limit did not demonstrate the same advantage}",
        r"\begin{table}[ht]\centering\small\begin{tabular}{llrrr}\toprule",
        r"Model & Seed & 2K on same 64 (\%) & 4K on 64 (\%) & Difference (pp) \\",
        r"\midrule",
    ]
    for model in models:
        arm = "Unchanged" if model == "base" else ("Answer only" if model.startswith("answer") else "Teacher trace")
        seed = "--" if model == "base" else model.split("_")[1]
        lines.append(
            f"{arm} & {seed} & {acc(short[model]):.2f} & {acc(long[model]):.2f} & {100 * length_checks[model]['delta']:+.2f} "
            + r"\\"
        )
    lines += [
        r"\bottomrule\end{tabular}\caption{Paired descriptive length check. Every entry uses the same preselected 64 questions; effective batch sizes are 32 at 2K and 16 at 4K. Differences are computed before rounding.}\label{tab:length}\end{table}",
    ]
    long_contrast = baseline_long_checks["trace"]
    ll, lh = [100 * v for v in long_contrast["paired_item_bootstrap_95"]]
    lines.append(
        f"On the 64-question check, the unchanged student rose from {acc(short['base']):.2f}\\% at 2K to "
        f"{acc(long['base']):.2f}\\% at 4K. Trace SFT averaged "
        f"{np.mean([acc(short[f'trace_{s}']) for s in SEEDS]):.2f}\\% and "
        f"{np.mean([acc(long[f'trace_{s}']) for s in SEEDS]):.2f}\\%, respectively "
        "(Table~\\ref{tab:length}). The descriptive 4K trace-minus-unchanged difference was "
        f"{100 * long_contrast['mean_delta']:+.2f} points (conditional item interval {ll:+.2f} to {lh:+.2f}). "
        "This check did not demonstrate an advantage over the unchanged student at 4K. Its small sample and changed batch "
        "size prevent treating the result as a general reversal, an equivalence test or a pure isolation of output length."
    )
    Path("paper/distillation_results.tex").write_text("\n".join(lines) + "\n")

    plt.rcParams.update(
        {"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6))
    x = np.arange(3)
    axes[0].plot(x, [acc(main[f"answer_{s}"]) for s in SEEDS], "o-", label="Answer only", color="#3578a8")
    axes[0].plot(x, [acc(main[f"trace_{s}"]) for s in SEEDS], "o-", label="Teacher trace", color="#bd6a2e")
    axes[0].axhline(acc(main["base"]), color="#555555", linestyle="--", label="Unchanged")
    axes[0].set(
        xticks=x, xticklabels=list(SEEDS), xlabel="Training seed", ylabel="Numeric accuracy (%)",
        title="256 questions at 2K", ylim=(0, 100),
    )
    axes[0].legend(frameon=False, fontsize=11)
    for index, seed in enumerate(SEEDS):
        axes[1].plot(
            [0, 1], [acc(short[f"trace_{seed}"]), acc(long[f"trace_{seed}"])],
            "o-", color="#d5b59a", linewidth=1, markersize=4,
            label="Trace seeds" if index == 0 else None,
        )
    axes[1].plot(
        [0, 1], [np.mean([acc(short[f"trace_{s}"]) for s in SEEDS]), np.mean([acc(long[f"trace_{s}"]) for s in SEEDS])],
        "o-", color="#bd6a2e", linewidth=2, label="Trace mean",
    )
    axes[1].plot([0, 1], [acc(short["base"]), acc(long["base"])], "o--", color="#555555", label="Unchanged")
    axes[1].set(
        xticks=[0, 1], xticklabels=["2K", "4K"], xlim=(-0.15, 1.15), ylim=(65, 100),
        xlabel="Output limit", ylabel="Numeric accuracy (%)", title="Same 64 questions at both limits",
    )
    axes[1].legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()
    fig.savefig("assets/distillation-2026-10-20.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    atomic_json(
        Path("paper/distillation-render.json"),
        {
            "protocol_id": protocol["protocol_id"],
            "generator_sha256": file_hash(Path(__file__)),
            "analysis_sha256": file_hash(proposal / "collected/analysis.json"),
            "secondary_analysis_sha256": file_hash(proposal / "collected/descriptive-secondary-analysis.json"),
            "figure_sha256": file_hash(Path("assets/distillation-2026-10-20.png")),
            "results_tex_sha256": file_hash(Path("paper/distillation_results.tex")),
            "abstract_tex_sha256": file_hash(Path("paper/distillation_abstract_results.tex")),
        },
    )
    print(
        json.dumps(
            {
                "primary": primary,
                "base_comparisons": baseline_checks,
                "secondary": str(proposal / "collected/descriptive-secondary-analysis.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    generate(parser.parse_args().proposal)
