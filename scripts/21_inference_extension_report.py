"""Render completed sampled-inference results and original completion diagnostics on CPU."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.artifacts import atomic_json, file_hash, fingerprint
from src.inference_extension import BUDGETS, MODELS, ORIGINAL, contrast, score_rows, summarize


def pct(value):
    return f"{100 * value:.2f}" if value is not None else "--"


def generate(proposal):
    proposal = Path(proposal)
    analysis = json.loads((proposal / "collected/analysis.json").read_text())
    protocol = json.loads((proposal / "protocol.json").read_text())
    if analysis["status"] != "complete" or analysis["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("A complete matching follow-up is required")
    primary = analysis["primary_8192"]
    strict_rows = {
        m: [
            json.loads(x)
            for x in (proposal / "collected" / f"eval_8192_{m}" / "predictions.jsonl").read_text().splitlines()
        ]
        for m in MODELS
    }
    strict8 = contrast([strict_rows[m] for m in MODELS[1:]], [strict_rows["base"]] * 3)
    atomic_json(
        Path("paper/inference-extension-secondary.json"),
        {
            "protocol_id": protocol["protocol_id"],
            "endpoint": "boxed-final-v1",
            "status": "saved-output secondary analysis; numerical primary unchanged",
            "strict_8192": strict8,
            "raw_sha256": {
                m: file_hash(proposal / "collected" / f"eval_8192_{m}" / "predictions.jsonl") for m in MODELS
            },
        },
    )
    means = [np.mean([analysis["budgets"][str(b)]["arms"][m]["accuracy"] for m in MODELS[1:]]) for b in BUDGETS]
    base2 = analysis["budgets"]["2048"]["arms"]["base"]
    base8 = analysis["budgets"]["8192"]["arms"]["base"]
    trace2, trace8 = means[0], means[-1]
    trace2_tokens = np.mean([analysis["budgets"]["2048"]["arms"][m]["mean_generated_tokens"] for m in MODELS[1:]])
    trace8_tokens = np.mean([analysis["budgets"]["8192"]["arms"][m]["mean_generated_tokens"] for m in MODELS[1:]])
    trace_eos = {
        b: np.mean([analysis["budgets"][str(b)]["arms"][m]["eos_count"] / 128 for m in MODELS[1:]])
        for b in (2048, 8192)
    }
    off = analysis["no_thinking_diagnostics"]["eval:2048:nothink_sampled:base"]
    off_greedy = analysis["no_thinking_diagnostics"]["eval:2048:nothink_greedy:base"]
    old = analysis["original_greedy_same_subset"]
    old_answer_mean = np.mean([old[f"answer_{s}"]["accuracy"] for s in (42, 43, 44)])
    lower, upper = primary["paired_item_bootstrap_95"]
    if lower > 0:
        verdict = (
            "The conditional item interval supports a positive numerical accuracy difference at 8K on this sample."
        )
    elif upper < 0:
        verdict = (
            "The conditional item interval supports a negative numerical accuracy difference at 8K on this sample."
        )
    else:
        verdict = "The 8K numerical comparison establishes neither an accuracy advantage nor equivalence."
    plan = json.loads((proposal / "plan.json").read_text())
    quote = json.loads((proposal / "quote.json").read_text())
    ledger_path = Path("outputs/inference-extension") / fingerprint(plan)[:24] / "ledger.json"
    if not ledger_path.exists():
        ledger_path = proposal / "execution-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if ledger["status"] != "complete" or ledger["identity"]["plan"] != plan:
        raise ValueError("Cost reporting requires the complete matching execution ledger")
    elapsed_cost = sum(
        v["elapsed_seconds"] * quote["tasks"][k]["rate_per_second"] for k, v in ledger["results"].items()
    )
    startup = sum(v["overhead_seconds"] * v["rate_per_second"] for v in quote["tasks"].values())
    gpu_hours = sum(ledger["results"][n["id"]]["elapsed_seconds"] for n in plan["nodes"] if n["gpu"]) / 3600
    cost = {
        "client_elapsed_at_resource_caps_usd": elapsed_cost,
        "extra_startup_allowance_usd": startup,
        "plus_startup_allowance_usd": elapsed_cost + startup,
        "plus_startup_and_incidental_allowances_usd": elapsed_cost + startup + quote["non_runtime_allowance_usd"],
        "gpu_call_hours": gpu_hours,
        "cap_usd": 30,
        "ledger_sha256": file_hash(ledger_path),
        "method": "Elapsed time times bounded per-task resource rates; extra startup allowance can overlap measured elapsed time; not a reconciled bill",
    }
    atomic_json(Path("paper/inference-extension-cost.json"), cost)
    Path("paper/inference_extension_cost.tex").write_text(
        f"The inference follow-up used {gpu_hours:.2f} aggregate L40S GPU-call hours across parallel evaluations. "
        f"Elapsed time priced at the configured GPU, CPU and memory rates gives an estimated \\${elapsed_cost:.2f}; "
        f"including extra startup and incidental allowances gives \\${elapsed_cost + startup + quote['non_runtime_allowance_usd']:.2f}. "
        "The original training-and-evaluation study's corresponding resource-time estimate was \\$7.01. "
        "These estimates exclude upstream generation of the public teacher data and are not reconciled provider invoices. "
        "Rates were checked against \\url{https://modal.com/pricing} on October 20, 2026.\n"
    )
    lines = [
        r"\section{Sampled inference results}",
        r"\label{sec:followup}",
        r"\subsection{Accuracy across output limits}",
        (
            f"Trace checkpoints averaged {pct(trace2)}\\% accuracy at 2K, compared with {pct(base2['accuracy'])}\\% "
            f"for the unchanged student, a descriptive difference of {100 * (trace2 - base2['accuracy']):+.2f} "
            f"percentage points. The corresponding differences at 4K and 8K were "
            f"{100 * analysis['budgets']['4096']['trace_minus_unchanged']['mean_delta']:+.2f} and "
            f"{100 * primary['mean_delta']:+.2f} points (Table~\\ref{{tab:followup-accuracy}})."
        ),
        r"\par\begin{table}[ht]\centering\small\begin{tabular}{llrrr}\toprule",
        r"Checkpoint & Training seed & 2K (\%) & 4K (\%) & 8K (\%) \\",
        r"\midrule",
    ]
    for model in MODELS:
        arm, seed = ("Unchanged", "--") if model == "base" else ("Teacher trace", model.split("_")[1])
        cells = [pct(analysis["budgets"][str(b)]["arms"][model]["accuracy"]) for b in BUDGETS]
        lines.append(f"{arm} & {seed} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            "Trace mean & -- & " + " & ".join(pct(x) for x in means) + r" \\",
            "Trace mean $-$ unchanged & -- & "
            + " & ".join(
                f"{100 * analysis['budgets'][str(b)]['trace_minus_unchanged']['mean_delta']:+.2f}" for b in BUDGETS
            )
            + r" \\",
            r"\bottomrule\end{tabular}\caption{Numerical primary accuracy on the same 128 questions at each output limit. The trace mean averages three trained checkpoints; the last row gives percentage-point differences. Thinking-mode sampling uses one fixed decoding seed and effective batches of 32/16/8.}\label{tab:followup-accuracy}\end{table}\par",
            (
                f"At the follow-up's primary 8K limit, trace checkpoints averaged {pct(trace8)}\\% versus "
                f"{pct(base8['accuracy'])}\\% unchanged. The mean difference was {100 * primary['mean_delta']:+.2f} "
                f"percentage points (95\\% paired item-bootstrap interval {100 * primary['paired_item_bootstrap_95'][0]:+.2f} "
                f"to {100 * primary['paired_item_bootstrap_95'][1]:+.2f}); the training-seed difference SD was "
                f"{100 * primary['training_seed_delta_std']:.2f} points. The interval conditions on the three trained "
                f"checkpoints and the one decoding seed. {verdict} Comparisons at 2K/4K are descriptive."
            ),
            r"\par\begin{figure}[ht]\centering\includegraphics[width=\textwidth]{assets/inference-extension-2026-10-20.png}",
            r"\caption{Sampled follow-up on 128 shared questions. Left: numerical accuracy; the blue square is the unchanged student's sampled no-thinking policy, tested only at 2K. Right: fraction terminating at EOS rather than the output limit. Thin orange lines are individual trace checkpoints; thicker lines show their mean. Effective batch size changes with the limit.}\label{fig:followup}\end{figure}\par",
            r"\subsection{Sensitivity to final-answer formatting}",
            (
                "Five unchanged-student responses at 8K contain repeated, numerically consistent correct boxes. "
                "They pass the numerical primary rule and fail the exactly-one-box secondary rule. The trace scores "
                "are unchanged. The strict rule therefore reverses the point estimate "
                f"to {100 * strict8['mean_delta']:+.2f} points, with a conditional item interval of "
                f"{100 * strict8['paired_item_bootstrap_95'][0]:+.2f} to {100 * strict8['paired_item_bootstrap_95'][1]:+.2f} "
                "(Table~\\ref{tab:format-sensitivity}). Neither rule establishes an 8K advantage. The primary "
                "rule was fixed before the follow-up generations and is retained throughout."
            ),
            r"\par\begin{table}[ht]\centering\small\begin{tabular}{lrrr}\toprule",
            r"8K scoring rule & Unchanged (\%) & Trace mean (\%) & Difference (pp) \\",
            r"\midrule",
            f"Numerical, primary & {pct(base8['accuracy'])} & {pct(trace8)} & {100 * primary['mean_delta']:+.2f} "
            + r"\\",
            f"One box, secondary & {pct(base8['strict_accuracy'])} & {pct(trace8)} & {100 * strict8['mean_delta']:+.2f} "
            + r"\\",
            r"\bottomrule\end{tabular}\caption{The same 128 questions and raw generations, scored under two prospectively retained rules. The numerical primary accepts repeated identical numeric boxes; the strict secondary requires exactly one box. Differences are trace mean minus unchanged.}\label{tab:format-sensitivity}\end{table}\par",
            r"\subsection{Completion and response length}",
            (
                f"At 2K, {100 * base2['eos_count'] / 128:.2f}\\% of unchanged-student responses ended at an "
                f"end-of-sequence (EOS) token, compared with {pct(trace_eos[2048])}\\% across trace checkpoints. "
                f"Mean generated lengths were {base2['mean_generated_tokens']:.0f} and {trace2_tokens:.0f} tokens, "
                f"respectively. At 8K, EOS rates were {100 * base8['eos_count'] / 128:.2f}\\% and "
                f"{pct(trace_eos[8192])}\\%; trace checkpoints still generated fewer tokens on average "
                f"({trace8_tokens:.0f} versus {base8['mean_generated_tokens']:.0f}; Table~\\ref{{tab:followup-completion}}). "
                "These lengths include all generated thinking and final-response tokens, subject to the stated limits."
            ),
            r"\par\begin{table}[ht]\centering\small\begin{tabular}{llrrrr}\toprule",
            r"Checkpoint & Seed & Correct/128 & EOS (\%) & Unfinished (\%) & Tokens \\",
            r"\midrule",
        ]
    )
    for model in MODELS:
        arm, seed = ("Unchanged", "--") if model == "base" else ("Teacher trace", model.split("_")[1])
        s = analysis["budgets"]["8192"]["arms"][model]
        lines.append(
            f"{arm} & {seed} & {s['num_correct']} & {100 * s['eos_count'] / 128:.2f} & "
            f"{100 * s['unfinished_thinking_count'] / 128:.2f} & {s['mean_generated_tokens']:.0f} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule\end{tabular}\caption{Completion and mean generated tokens at 8K. Unfinished denotes an unclosed thinking block, which is distinct from EOS termination. Missing or invalid final fields remain incorrect.}\label{tab:followup-completion}\end{table}\par",
            r"\subsection{Native no-thinking inference and the failed training control}",
            (
                f"Disabling the unchanged student's thinking channel yielded {pct(off_greedy['accuracy'])}\\% "
                f"accuracy with greedy decoding and {pct(off['accuracy'])}\\% with its recommended sampled policy. "
                f"The original answer-only checkpoints averaged {pct(old_answer_mean)}\\% under greedy decoding "
                "on these exact 128 questions. Native no-thinking therefore did not reproduce the answer-only "
                "SFT collapse (Table~\\ref{tab:nothinking}). Both native no-thinking runs terminated at EOS on all 128 questions."
            ),
            r"\par\begin{table}[ht]\centering\small\begin{tabular}{llrr}\toprule",
            r"Checkpoint or recipe & Decoding & Accuracy (\%) & Mean tokens \\",
            r"\midrule",
        ]
    )
    for policy in ("greedy", "sampled"):
        s = analysis["no_thinking_diagnostics"][f"eval:2048:nothink_{policy}:base"]
        lines.append(
            f"Unchanged, no thinking & {policy} & {pct(s['accuracy'])} & {s['mean_generated_tokens']:.0f} " + r"\\"
        )
    lines.append(
        f"Unchanged, thinking & greedy & {pct(old['base']['accuracy'])} & {old['base']['mean_generated_tokens']:.0f} "
        + r"\\"
    )
    for seed in (42, 43, 44):
        s = old[f"answer_{seed}"]
        lines.append(
            f"Answer-only SFT, seed {seed} & greedy & {pct(s['accuracy'])} & {s['mean_generated_tokens']:.0f} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule\end{tabular}\caption{All entries use the same 128 questions at 2K. Greedy thinking and answer-only entries are retained original outputs on this subset, not reruns. The no-thinking sampled policy uses temperature 0.7 and top-p 0.8; it must not be treated as decoder-matched to greedy outputs.}\label{tab:nothinking}\end{table}\par",
            (
                f"The sampled native policy used {off['mean_generated_tokens']:.0f} mean tokens, compared with "
                f"{trace2_tokens:.0f} for trace checkpoints at 2K, with accuracies of {pct(off['accuracy'])}\\% and "
                f"{pct(trace2)}\\%. This is a descriptive comparison of complete inference recipes: thinking and "
                "no-thinking use different sampling parameters. Disabling the explicit thinking channel also "
                "permits worked steps in the final response; it does not imply an absence of reasoning."
            ),
        ]
    )
    Path("paper/inference_extension_results.tex").write_text("\n".join(lines) + "\n")

    abstract = (
        f"At 2K, trace checkpoints averaged {pct(trace2)}\\% numerical accuracy versus {pct(base2['accuracy'])}\\% "
        f"unchanged. At the follow-up's primary 8K limit, the corresponding accuracies were {pct(trace8)}\\% and "
        f"{pct(base8['accuracy'])}\\% (difference {100 * primary['mean_delta']:+.2f} percentage points; "
        f"95\\% conditional paired item interval {100 * lower:+.2f} to {100 * upper:+.2f}). "
        "The one-box secondary rule reverses the 8K point estimate, but its interval also includes zero. "
        f"The unchanged student's sampled no-thinking policy reached {pct(off['accuracy'])}\\% at 2K with "
        f"{off['mean_generated_tokens']:.0f} mean tokens, versus {trace2_tokens:.0f} for trace checkpoints. "
        "Native no-thinking did not reproduce the original answer-only SFT collapse.\n"
    )
    Path("paper/inference_extension_abstract.tex").write_text(abstract)
    Path("paper/inference_extension_overview.tex").write_text(
        f"The trace checkpoints had higher numerical accuracy at 2K ({pct(trace2)}\\% versus "
        f"{pct(base2['accuracy'])}\\%) and generated shorter responses. At 8K, where almost all responses "
        f"terminated at an end-of-sequence (EOS) token, their mean was {pct(trace8)}\\% versus {pct(base8['accuracy'])}\\% unchanged. "
        "The conditional item interval includes zero under both retained scoring rules. Native no-thinking "
        "inference also avoided the severe failure seen after answer-only SFT. Together, these results "
        "show how output allowances, final-answer formatting and an existing inference mode change "
        "the interpretation of a small trace-distillation experiment.\n"
    )
    interpretation = (
        "The trace checkpoints' shorter responses are accompanied by a numerical accuracy advantage at "
        "the 2K limit, where the unchanged student often fails to terminate. At 4K and 8K its numerical "
        "point estimate is higher. This pattern, together with the original-output diagnostic, supports "
        "completion behavior as an important part of the short-limit result. It does not identify how "
        "a truncated response would have answered if continued.\n\n"
        f"Long supervision also coexists with shorter student outputs. The serialized trace-arm targets average "
        f"{2635116 / 500:.0f} supervised tokens per example; at 8K the students generate {trace8_tokens:.0f} "
        f"tokens on average, versus {base8['mean_generated_tokens']:.0f} unchanged. The training and test tasks "
        "differ, and teacher length was not varied. These observations do not establish that the student "
        "learned to compress the teacher's reasoning.\n\n"
        "The no-thinking controls change the interpretation of the failed answer-only arm. Switching off "
        "the native thinking channel is insufficient to reproduce its collapse. They also provide a "
        "practical baseline for short-output use: the sampled native policy's observed accuracy is close "
        "to the 2K trace mean and its responses are much shorter. The policy comparison remains "
        "descriptive; token counts alone do not measure latency or proportional GPU-cost savings.\n"
    )
    Path("paper/inference_extension_interpretation.tex").write_text(interpretation)
    conclusion = (
        "In this Qwen3-1.7B case study, trace SFT produced shorter observed GSM8K responses and higher "
        f"numerical accuracy at the 2K limit. {verdict} The strict one-box secondary rule changes the "
        "8K point estimate's sign without establishing an advantage. Native no-thinking inference "
        "did not reproduce the answer-only SFT collapse. Evaluating trace distillation in this setting "
        "therefore requires output-budget comparisons, explicit scoring rules and native inference controls "
        "alongside the trained checkpoints.\n"
    )
    Path("paper/inference_extension_conclusion.tex").write_text(conclusion)

    old_analysis = json.loads((ORIGINAL / "collected/analysis.json").read_text())
    old_primary = old_analysis["primary_2048"]["primary"]
    initial = [
        r"\par\begin{table}[ht]\centering\small\begin{tabular}{llrrr}\toprule",
        r"Checkpoint & Seed & Correct/256 & Accuracy (\%) & Tokens \\",
        r"\midrule",
    ]
    for model in ["base", *[f"{arm}_{s}" for s in (42, 43, 44) for arm in ("answer", "trace")]]:
        s = old_analysis["primary_2048"]["arms"][model]
        name = "Unchanged" if model == "base" else ("Answer only" if model.startswith("answer") else "Teacher trace")
        seed = "--" if model == "base" else model.split("_")[1]
        initial.append(
            f"{name} & {seed} & {round(s['accuracy'] * 256)} & {pct(s['accuracy'])} & {s['mean_generated_tokens']:.0f} "
            + r"\\"
        )
    initial += [
        r"\bottomrule\end{tabular}\caption{Original numerical outcomes on 256 common GSM8K questions with greedy thinking and a 2K limit. All seven checkpoints use the same questions.}\label{tab:original}\end{table}\par",
        (
            f"The original primary trace-minus-answer difference was {100 * old_primary['mean_delta']:+.2f} "
            f"percentage points (95\\% conditional item interval {100 * old_primary['paired_item_bootstrap_95'][0]:+.2f} "
            f"to {100 * old_primary['paired_item_bootstrap_95'][1]:+.2f}; seed-difference SD "
            f"{100 * old_primary['training_seed_delta_std']:.2f} points). Trace SFT averaged 78.12\\%, "
            "answer-only SFT 6.25\\% and the unchanged student 70.31\\%. The separate descriptive trace-minus-unchanged "
            "difference was +7.81 points (conditional interval +3.39 to +12.50). The unchanged student's strict score "
            "was 176/256; strict and numerical scores coincide for the other six checkpoints.\n"
        ),
        (
            "All 768 answer-only outputs contained empty visible thinking blocks. The recipe's large deficit limits "
            "interpretation of the primary contrast. An original preselected 64-question greedy check at 4K yielded "
            "a trace mean of 83.85\\% versus 87.50\\% unchanged, a descriptive difference of $-3.65$ points "
            "(conditional interval $-8.33$ to $+1.04$). This check uses a different subset and greedy decoding, "
            "rather than the follow-up's sampled policy.\n"
        ),
    ]
    Path("paper/original_results_summary.tex").write_text("\n".join(initial).rstrip() + "\n")

    original_rows = {}
    for model in MODELS:
        raw = [
            json.loads(x) for x in (ORIGINAL / "collected/2048" / model / "predictions.jsonl").read_text().splitlines()
        ]
        original_rows[model] = score_rows(raw)
    original_stats = {m: summarize(rows) for m, rows in original_rows.items()}
    pooled = summarize([r for m, rr in original_rows.items() if m != "base" for r in rr])
    pooled_rows = [r for m, rr in original_rows.items() if m != "base" for r in rr]
    base = {r["problem_id"]: r for r in original_rows["base"]}
    trace_only, base_limited, base_unfinished = 0, 0, 0
    base_only, trace_limited = 0, 0
    for model in MODELS[1:]:
        for row in original_rows[model]:
            other = base[row["problem_id"]]
            if row["correct"] and not other["correct"]:
                trace_only += 1
                base_limited += other["finish_reason"] == "length"
                base_unfinished += other["status"] == "unfinished_thinking"
            elif other["correct"] and not row["correct"]:
                base_only += 1
                trace_limited += row["finish_reason"] == "length"
    diagnostic = {
        "original_256": original_stats,
        "trace_pooled": pooled,
        "trace_only_correct_question_seed_pairs": trace_only,
        "of_these_unchanged_hit_output_limit": base_limited,
        "of_these_unchanged_unfinished_thinking": base_unfinished,
        "unchanged_only_correct_question_seed_pairs": base_only,
        "of_these_trace_hit_output_limit": trace_limited,
        "caution": "Post hoc diagnostics; question-seed pairs reuse questions; conditional subsets are selected by each model's behavior",
    }
    atomic_json(Path("paper/original-completion-diagnostic.json"), diagnostic)
    original_tex = [
        r"\subsection{Post hoc completion diagnostic for the original 2K evaluation}",
        (
            f"Among the {trace_only} question--seed pairs where a trace checkpoint was correct and the unchanged student was "
            f"incorrect, the unchanged response hit the output limit in {base_limited} pairs. These pairs reuse the same "
            f"256 questions across training seeds. Conversely, there were {base_only} unchanged-only correct pairs, "
            f"of which {trace_limited} involved an output-limited trace response. Most trace-only correct pairs "
            "thus involve an output-limited unchanged response, while leaving unknown how those responses "
            "would have answered with more generation."
        ),
        r"\par\begin{table}[ht]\centering\small\begin{tabular}{lrr}\toprule",
        r"Original 2K subset & Unchanged & Trace, pooled \\",
        r"\midrule",
    ]
    for label, selected in (
        ("EOS-terminated", lambda row: row["finish_reason"] == "eos"),
        ("No unfinished thinking", lambda row: row["status"] != "unfinished_thinking"),
    ):
        cells = []
        for rows in (original_rows["base"], pooled_rows):
            subset = [row for row in rows if selected(row)]
            correct = sum(row["correct"] for row in subset)
            cells.append(f"{correct}/{len(subset)} ({pct(correct / len(subset))}\\%)")
        original_tex.append(f"{label} & " + " & ".join(cells) + r" \\")
    original_tex.append(
        r"\bottomrule\end{tabular}\caption{Correct responses divided by each model-dependent subset, with conditional accuracy in parentheses. Trace pooling reuses 256 questions across three checkpoints. Closing a thinking block does not imply EOS termination; these selected subsets do not isolate reasoning ability.}\label{tab:original-completion}\end{table}\par"
    )
    Path("paper/original_completion_diagnostic.tex").write_text("\n".join(original_tex) + "\n")

    plt.rcParams.update(
        {"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6))
    for ax, metric in zip(axes, ("accuracy", "eos"), strict=True):

        def values(model, metric=metric):
            arms = [analysis["budgets"][str(b)]["arms"][model] for b in BUDGETS]
            return [100 * (s["accuracy"] if metric == "accuracy" else s["eos_count"] / 128) for s in arms]

        for i, model in enumerate(MODELS[1:]):
            ax.plot(
                range(3),
                values(model),
                "o-",
                color="#d5b59a",
                linewidth=1,
                markersize=4,
                label="Trace checkpoints" if i == 0 else None,
            )
        ax.plot(
            range(3),
            np.mean([values(m) for m in MODELS[1:]], axis=0),
            "o-",
            color="#bd6a2e",
            label="Trace mean",
            linewidth=2,
        )
        ax.plot(range(3), values("base"), "o--", color="#555555", label="Unchanged")
        if metric == "accuracy":
            ax.scatter(
                [0], [100 * off["accuracy"]], marker="s", color="#3578a8", zorder=4, label="Unchanged, no thinking (2K)"
            )
        ax.set(
            xticks=range(3),
            xticklabels=["2K", "4K", "8K"],
            xlabel="Output limit",
            ylim=(0, 103),
            ylabel="Numerical accuracy (%)" if metric == "accuracy" else "EOS termination (%)",
        )
        ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig("assets/inference-extension-2026-10-20.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    atomic_json(
        Path("paper/inference-extension-render.json"),
        {
            "protocol_id": protocol["protocol_id"],
            "analysis_sha256": file_hash(proposal / "collected/analysis.json"),
            "generator_sha256": file_hash(Path(__file__)),
            "figure_sha256": file_hash(Path("assets/inference-extension-2026-10-20.png")),
            "results_tex_sha256": file_hash(Path("paper/inference_extension_results.tex")),
            "diagnostic_json_sha256": file_hash(Path("paper/original-completion-diagnostic.json")),
            "secondary_json_sha256": file_hash(Path("paper/inference-extension-secondary.json")),
        },
    )
    report = [
        "# Sampled inference follow-up results — October 20, 2026",
        "",
        (
            "The frozen inference-only package completed 14 GPU evaluations and 14 separate persistence checks, "
            "after CPU preflight. It reused the original student and three trace checkpoints; no training was run. "
            "All 1,792 new raw responses passed hash, identity and CPU strict-score checks. The original experiment remains separate."
        ),
        "",
        "## Primary 8K comparison",
        "",
        (
            f"Trace checkpoints average **{pct(trace8)}%**, versus **{pct(base8['accuracy'])}%** unchanged. The mean difference "
            f"is **{100 * primary['mean_delta']:+.2f} percentage points**, with a 95% conditional paired item-bootstrap "
            f"interval of **{100 * lower:+.2f} to {100 * upper:+.2f}** and seed-difference SD "
            f"{100 * primary['training_seed_delta_std']:.2f} points. {verdict}"
        ),
        "",
        (
            "The interval conditions on three trained checkpoints and one decoding seed. There are 128 common questions, "
            "not 384 independent observations. The follow-up was designed after seeing the original findings and reuses "
            "previously evaluated questions. Eight thousand output tokens is still a constrained inference setting."
        ),
        "",
        "## Secondary sensitivity to answer formatting",
        "",
        (
            f"The unchanged student's strict one-box accuracy at 8K is **{pct(base8['strict_accuracy'])}%**, "
            f"versus **{pct(trace8)}%** for the trace mean. The strict trace-minus-unchanged difference is "
            f"**{100 * strict8['mean_delta']:+.2f} points** (95% conditional item interval "
            f"**{100 * strict8['paired_item_bootstrap_95'][0]:+.2f} to {100 * strict8['paired_item_bootstrap_95'][1]:+.2f}**). "
            "Five unchanged-student responses have repeated, numerically consistent correct boxes: they pass "
            "the numerical primary rule and fail the strict rule. The point estimate changes sign, while neither "
            "rule establishes an 8K advantage. This saved-output secondary analysis does not replace the primary endpoint."
        ),
        "",
        "## Same-question sampled results",
        "",
        "| Checkpoint | 2K accuracy | 4K accuracy | 8K accuracy | 8K EOS /128 | 8K mean tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        ss = [analysis["budgets"][str(b)]["arms"][model] for b in BUDGETS]
        report.append(
            f"| {model} | {pct(ss[0]['accuracy'])}% | {pct(ss[1]['accuracy'])}% | "
            f"{pct(ss[2]['accuracy'])}% | {ss[2]['eos_count']} | {ss[2]['mean_generated_tokens']:.1f} |"
        )
    report += [
        "",
        (
            "Thinking-mode sampling uses temperature 0.6, top-p 0.95, top-k 20 and disabled min-p, with seed 20261023. "
            "Effective batches are 32/16/8 across the limits. Canonical batch membership changes the RNG streams, "
            "so these are separately sampled trajectories, not truncations of a common rollout. The curve does "
            "not isolate a causal output-length effect. EOS, unfinished thinking and final-field failures are distinct."
        ),
        "",
        "## No-thinking controls and the answer-only collapse",
        "",
        "| Original student at 2K, thinking disabled | Correct /128 | Numerical accuracy | Mean tokens |",
        "|---|---:|---:|---:|",
    ]
    for policy in ("greedy", "sampled"):
        s = analysis["no_thinking_diagnostics"][f"eval:2048:nothink_{policy}:base"]
        report.append(f"| {policy} | {s['num_correct']} | {pct(s['accuracy'])}% | {s['mean_generated_tokens']:.1f} |")
    report += [
        "",
        (
            f"On precisely these 128 questions, the original answer-only checkpoints average **{pct(old_answer_mean)}%** "
            f"under greedy decoding. The original unchanged thinking-mode greedy score is **{pct(old['base']['accuracy'])}%**. "
            "Switching off the native thinking channel therefore does not reproduce the answer-only recipe's collapse. "
            "The control does not identify knowledge forgetting or the cause of training failure; native no-thinking "
            "outputs can still contain worked steps in the final response."
        ),
        "",
        (
            f"The sampled no-thinking policy uses temperature 0.7/top-p 0.8, while sampled thinking uses 0.6/0.95. "
            f"Its 2K accuracy is {pct(off['accuracy'])}% with {off['mean_generated_tokens']:.1f} mean generated tokens, "
            f"versus the trace mean of {pct(trace2)}% and {trace2_tokens:.1f} tokens. This compares complete model-and-policy "
            "recipes and is descriptive; one run does not establish equivalence or general policy ordering."
        ),
        "",
        "## Original 256-question completion diagnostic",
        "",
        (
            f"This new post hoc diagnostic uses only saved original outputs: {base_limited} of {trace_only} "
            "trace-only-correct question–seed pairs involve an unchanged-student response hitting the 2K limit. "
            f"Conversely, {trace_limited} of {base_only} unchanged-only-correct pairs involve a trace response hitting "
            "the limit. Those pairs reuse questions across seeds. Among outputs without unfinished thinking, accuracy is "
            f"{pct(original_stats['base']['accuracy_among_without_unfinished_thinking'])}% unchanged versus "
            f"{pct(pooled['accuracy_among_without_unfinished_thinking'])}% pooled trace. With actual EOS termination, "
            f"the corresponding rates are {pct(original_stats['base']['accuracy_among_eos'])}% and "
            f"{pct(pooled['accuracy_among_eos'])}%. Conditional subsets select different questions and cannot isolate reasoning ability."
        ),
        "",
        "## Cost and limits",
        "",
        (
            f"New client elapsed time priced at resource caps is estimated at **${elapsed_cost:.2f}** "
            f"({gpu_hours:.2f} aggregate GPU-call hours). Adding separate startup allowance gives "
            f"**${elapsed_cost + startup:.2f}**; adding the incidental allowance gives "
            f"**${elapsed_cost + startup + quote['non_runtime_allowance_usd']:.2f}**. The latter allowances can overlap "
            "observed time or remain unused. These are not reconciled provider invoices. Frozen reservations totaled "
            "$29.0453128 against the separate $30 cap. Credits and public teacher-generation costs are excluded from "
            "the budget justification and student-run accounting, respectively."
        ),
        "",
        (
            "One student, one training sample, one benchmark, one decoding seed and changing batch size limit the "
            "conclusions. No general reasoning-capacity gain, causal compression mechanism or publication acceptance "
            "is established. Broader training and benchmark extensions were not run."
        ),
        "",
        "## Evidence",
        "",
        f"- Frozen proposal: `{proposal}`.",
        f"- [Completed analysis](../../{proposal}/collected/analysis.json), with per-job raw hashes and both scoring endpoints.",
        "- [Paper source](../../paper/paper.tex), [PDF](../../assets/paper.pdf) and [follow-up figure](../../assets/inference-extension-2026-10-20.png).",
        "- [Compute estimate](../../paper/inference-extension-cost.json) and [original completion diagnostic](../../paper/original-completion-diagnostic.json).",
        "- [Strict secondary 8K analysis](../../paper/inference-extension-secondary.json), derived from the same saved raw outputs.",
        "- [Protocol and execution scope](inference-extension-2026-10-20.md).",
    ]
    report_text = "\n".join(report) + "\n"
    Path("docs/experiments/inference-extension-results-2026-10-20.md").write_text(report_text)
    print(json.dumps({"primary_8192": primary, "original_completion_diagnostic": diagnostic}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True)
    generate(parser.parse_args().proposal)
