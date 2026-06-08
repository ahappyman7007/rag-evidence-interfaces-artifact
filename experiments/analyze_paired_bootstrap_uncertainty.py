#!/usr/bin/env python3
"""Paired bootstrap uncertainty for matched evidence-interface comparisons.

The script reads existing prediction JSONL files referenced by rank-sweep
summary files. It does not run model inference. Each comparison is paired by
`source_id`, then bootstrapped over validation examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RANK_SWEEP = ROOT / "experiments" / "outputs" / "rank_sweep"
DEFAULT_OUT_DIR = ROOT / "experiments" / "outputs" / "consolidated" / "tables"


DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "2wiki": "2WikiMultiHopQA",
    "musique": "MuSiQue",
}

INTERFACE_LABELS = {
    "raw_context": "Raw context",
    "raw_support_first": "Support first",
    "raw_ftcrossenc_top5docs": "FT cross-enc top-5",
    "raw_crossenc_top5docs": "Cross-enc top-5",
    "gold_supporting_sentences": "Oracle support",
    "gold_supporting_paragraphs": "Oracle support",
    "gold_evidence_triples": "Structured triples",
}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    interface: str
    summary_file: str


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    dataset: str
    baseline: str
    comparison: str
    baseline_file: str
    comparison_file: str


MAIN_QWEN14B_N4000 = [
    ComparisonSpec(
        "support_first_minus_raw",
        "hotpotqa",
        "raw_context",
        "raw_support_first",
        "hotpotqa_qwen14b_r8_n4000_raw_context_summary.json",
        "hotpotqa_qwen14b_r8_n4000_raw_support_first_summary.json",
    ),
    ComparisonSpec(
        "top5_minus_raw",
        "hotpotqa",
        "raw_context",
        "raw_ftcrossenc_top5docs",
        "hotpotqa_qwen14b_r8_n4000_raw_context_summary.json",
        "hotpotqa_qwen14b_r8_n4000_raw_ftcrossenc_top5docs_summary.json",
    ),
    ComparisonSpec(
        "oracle_minus_raw",
        "hotpotqa",
        "raw_context",
        "gold_supporting_sentences",
        "hotpotqa_qwen14b_r8_n4000_raw_context_summary.json",
        "hotpotqa_qwen14b_r8_n4000_gold_supporting_sentences_summary.json",
    ),
    ComparisonSpec(
        "support_first_minus_raw",
        "2wiki",
        "raw_context",
        "raw_support_first",
        "2wiki_qwen14b_r8_n4000_raw_context_summary.json",
        "2wiki_qwen14b_r8_n4000_raw_support_first_summary.json",
    ),
    ComparisonSpec(
        "top5_minus_raw",
        "2wiki",
        "raw_context",
        "raw_crossenc_top5docs",
        "2wiki_qwen14b_r8_n4000_raw_context_summary.json",
        "2wiki_qwen14b_r8_n4000_raw_crossenc_top5docs_summary.json",
    ),
    ComparisonSpec(
        "oracle_minus_raw",
        "2wiki",
        "raw_context",
        "gold_supporting_sentences",
        "2wiki_qwen14b_r8_n4000_raw_context_summary.json",
        "2wiki_qwen14b_r8_n4000_gold_supporting_sentences_summary.json",
    ),
    ComparisonSpec(
        "structured_minus_raw",
        "2wiki",
        "raw_context",
        "gold_evidence_triples",
        "2wiki_qwen14b_r8_n4000_raw_context_summary.json",
        "2wiki_qwen14b_r8_n4000_gold_evidence_triples_summary.json",
    ),
    ComparisonSpec(
        "support_first_minus_raw",
        "musique",
        "raw_context",
        "raw_support_first",
        "musique_qwen14b_r8_n4000_raw_context_summary.json",
        "musique_qwen14b_r8_n4000_raw_support_first_summary.json",
    ),
    ComparisonSpec(
        "top5_minus_raw",
        "musique",
        "raw_context",
        "raw_crossenc_top5docs",
        "musique_qwen14b_r8_n4000_raw_context_summary.json",
        "musique_qwen14b_r8_n4000_raw_crossenc_top5docs_summary.json",
    ),
    ComparisonSpec(
        "oracle_minus_raw",
        "musique",
        "raw_context",
        "gold_supporting_paragraphs",
        "musique_qwen14b_r8_n4000_raw_context_summary.json",
        "musique_qwen14b_r8_n4000_gold_supporting_paragraphs_summary.json",
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prediction_path_from_summary(summary_path: Path, interface: str) -> Path:
    data = load_json(summary_path)
    rows = data.get("rows", [])
    matches = [row for row in rows if row.get("interface_name") == interface]
    if not matches and len(rows) == 1:
        matches = rows
    if not matches:
        raise ValueError(f"no row for {interface} in {summary_path}")
    eval_summary = Path(matches[0]["eval_summary_path"])
    pred_path = eval_summary.with_name("predictions.jsonl")
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    return pred_path


def load_metric_by_source_id(pred_path: Path, metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    with pred_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_id = str(row.get("source_id") or row.get("id"))
            metrics = row.get("metrics", {})
            if metric not in metrics:
                raise KeyError(f"{metric} missing in {pred_path}")
            values[source_id] = float(metrics[metric])
    return values


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty percentile input")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_difference(
    baseline_values: list[float],
    comparison_values: list[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if len(baseline_values) != len(comparison_values):
        raise ValueError("paired vectors must have the same length")
    n = len(baseline_values)
    diffs = [comparison_values[i] - baseline_values[i] for i in range(n)]
    observed = mean(diffs)
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _j in range(n):
            total += diffs[rng.randrange(n)]
        boot.append(total / n)
    boot.sort()
    gt_zero = sum(1 for value in boot if value > 0.0) / samples
    lt_zero = sum(1 for value in boot if value < 0.0) / samples
    return {
        "observed_diff": observed,
        "ci_low": percentile(boot, 0.025),
        "ci_high": percentile(boot, 0.975),
        "prob_diff_gt_zero": gt_zero,
        "prob_diff_lt_zero": lt_zero,
    }


def stable_seed_offset(text: str) -> int:
    """Small deterministic offset so each comparison gets its own stream."""
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % 100000


def analyze_comparison(spec: ComparisonSpec, metric: str, samples: int, seed: int) -> dict[str, Any]:
    baseline_pred = prediction_path_from_summary(RANK_SWEEP / spec.baseline_file, spec.baseline)
    comparison_pred = prediction_path_from_summary(RANK_SWEEP / spec.comparison_file, spec.comparison)
    baseline = load_metric_by_source_id(baseline_pred, metric)
    comparison = load_metric_by_source_id(comparison_pred, metric)
    paired_ids = sorted(set(baseline) & set(comparison))
    if not paired_ids:
        raise ValueError(f"no paired ids for {spec}")
    baseline_values = [baseline[source_id] for source_id in paired_ids]
    comparison_values = [comparison[source_id] for source_id in paired_ids]
    stats = bootstrap_difference(
        baseline_values,
        comparison_values,
        samples=samples,
        seed=seed + stable_seed_offset(f"{spec.dataset}:{spec.name}"),
    )
    return {
        "dataset": spec.dataset,
        "dataset_label": DATASET_LABELS[spec.dataset],
        "comparison_name": spec.name,
        "baseline_interface": spec.baseline,
        "baseline_label": INTERFACE_LABELS.get(spec.baseline, spec.baseline),
        "comparison_interface": spec.comparison,
        "comparison_label": INTERFACE_LABELS.get(spec.comparison, spec.comparison),
        "metric": metric,
        "n_paired": len(paired_ids),
        "baseline_mean": round(mean(baseline_values), 6),
        "comparison_mean": round(mean(comparison_values), 6),
        "diff": round(stats["observed_diff"], 6),
        "ci_low": round(stats["ci_low"], 6),
        "ci_high": round(stats["ci_high"], 6),
        "prob_diff_gt_zero": round(stats["prob_diff_gt_zero"], 6),
        "prob_diff_lt_zero": round(stats["prob_diff_lt_zero"], 6),
        "bootstrap_samples": samples,
        "baseline_predictions": str(baseline_pred.relative_to(ROOT)),
        "comparison_predictions": str(comparison_pred.relative_to(ROOT)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Paired Bootstrap Uncertainty",
        "",
        "Paired by validation `source_id`. The reported difference is",
        "`comparison - baseline` on per-example F1.",
        "",
        "|Dataset|Comparison|Baseline F1|Comparison F1|Diff|95% CI|P(diff > 0)|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["dataset_label"]),
                    f"{row['comparison_label']} - {row['baseline_label']}",
                    f"{float(row['baseline_mean']):.3f}",
                    f"{float(row['comparison_mean']):.3f}",
                    f"{float(row['diff']):+.3f}",
                    f"[{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}]",
                    f"{float(row['prob_diff_gt_zero']):.3f}",
                ]
            )
            + "|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--metric", default="f1", choices=["f1", "exact_match", "answer_contained"])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    rows = [
        analyze_comparison(spec, args.metric, args.samples, args.seed)
        for spec in MAIN_QWEN14B_N4000
    ]
    csv_path = args.out_dir / "ipm_paired_bootstrap_qwen14b_n4000.csv"
    md_path = args.out_dir / "ipm_paired_bootstrap_qwen14b_n4000.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
