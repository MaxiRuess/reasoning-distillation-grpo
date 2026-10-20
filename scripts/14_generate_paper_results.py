"""Render historical tables/figure from recorded paired analysis, never hand-edited scores."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def table(headers, rows, caption, columns):
    body = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}" + columns + "@{}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    body += [" & ".join(row) + r" \\" for row in rows]
    return "\n".join(body + [r"\bottomrule", r"\end{tabular}", "\\caption{" + caption + "}", r"\end{table}", ""])


def main():
    source = json.loads(Path("results/recovery/historical-paired-analysis.json").read_text())
    names = {
        "base_model": "Unchanged post-trained checkpoint",
        "baseline": "Orca worked-solution SFT",
        "sft_traces": "s1K reasoning-trace SFT",
        "grpo_only": "GRPO from starting checkpoint",
        "sft_then_grpo": r"s1K SFT then GRPO$^*$",
        "re_distill": "Open-R1 verified-trace SFT",
    }
    rows = []
    for name, label in names.items():
        result = source["conditions"][name]
        low, high = result["wilson_95"]
        rows.append(
            [
                label,
                f"{result['num_correct']:,}",
                f"{result['accuracy'] * 100:.2f}\\%",
                f"{low * 100:.2f}--{high * 100:.2f}\\%",
            ]
        )
    Path("paper/historical_counts.tex").write_text(
        table(
            ["Historical recipe", "Correct / 1,319", "Accuracy", r"95\% Wilson interval"],
            rows,
            r"Preserved historical outcomes under the legacy scorer and greedy decoding with a 2,048-token budget. These are not corrected or independently reproduced accuracies. $^*$The combined model requires reconstruction with its original SFT contribution.",
            "lrrl",
        )
    )
    comparisons = [
        ("re_distill__vs__sft_traces", "Open-R1 SFT minus s1K SFT"),
        ("grpo_only__vs__base_model", "GRPO minus starting checkpoint"),
        ("sft_then_grpo__vs__sft_traces", r"Combined$^*$ minus s1K SFT"),
        ("sft_traces__vs__base_model", "s1K SFT minus starting checkpoint"),
        ("re_distill__vs__grpo_only", "Open-R1 SFT minus GRPO"),
    ]
    labels = [label for _, label in comparisons]
    rows, effects, lows, highs = [], [], [], []
    for key, label in comparisons:
        result = source["comparisons"][key]
        low, high = result["paired_bootstrap_95"]
        delta = result["delta"]
        rows.append(
            [
                label,
                f"{delta * 100:+.2f}",
                f"{low * 100:+.2f} to {high * 100:+.2f}",
                f"{result['holm_adjusted_p_exploratory_family']:.5f}",
            ]
        )
        effects.append(delta * 100)
        lows.append((delta - low) * 100)
        highs.append((high - delta) * 100)
    Path("paper/historical_pairs.tex").write_text(
        table(
            ["Comparison", "Difference (pp)", r"Paired 95\% interval (pp)", r"Holm $p$"],
            rows,
            "Exploratory paired analysis of historical legacy-scored results. The combined-model caveat applies regardless of its statistical separation. Training-seed variability is unavailable for these runs.",
            "lrrr",
        )
    )
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    ax.axvline(0, color="#777777", linewidth=1)
    ax.errorbar(effects, range(len(labels)), xerr=[lows, highs], fmt="o", color="#24577a", capsize=4)
    ax.set_yticks(range(len(labels)), [label.replace("$^*$", "*") for label in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Difference in accuracy (percentage points); paired 95% bootstrap interval")
    ax.set_title("Historical outcome differences — legacy scoring", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig("assets/historical-paired-differences.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
