#!/usr/bin/env python3
"""Extract support-first win cases for a compact error taxonomy.

This script uses existing Qwen14B n=4000 prediction files. It does not run
model inference. The goal is to identify examples where raw context fails but
support-first succeeds, then attach reproducible diagnostic tags that make the
interface bottleneck easier to inspect.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RANK_SWEEP = ROOT / "experiments" / "outputs" / "rank_sweep"
DEFAULT_OUT_DIR = ROOT / "experiments" / "outputs" / "consolidated" / "tables"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    data_dir: Path
    raw_summary: str
    support_summary: str


DATASETS = [
    DatasetSpec(
        key="hotpotqa",
        label="HotpotQA",
        data_dir=ROOT / "data" / "hotpotqa_interfaces" / "pilot",
        raw_summary="hotpotqa_qwen14b_r8_n4000_raw_context_summary.json",
        support_summary="hotpotqa_qwen14b_r8_n4000_raw_support_first_summary.json",
    ),
    DatasetSpec(
        key="2wiki",
        label="2WikiMultiHopQA",
        data_dir=ROOT / "data" / "2wiki_interfaces" / "pilot",
        raw_summary="2wiki_qwen14b_r8_n4000_raw_context_summary.json",
        support_summary="2wiki_qwen14b_r8_n4000_raw_support_first_summary.json",
    ),
    DatasetSpec(
        key="musique",
        label="MuSiQue",
        data_dir=ROOT / "data" / "musique_interfaces" / "pilot",
        raw_summary="musique_qwen14b_r8_n4000_raw_context_summary.json",
        support_summary="musique_qwen14b_r8_n4000_raw_support_first_summary.json",
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl_by_source_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_id = str(row.get("source_id") or row.get("id"))
            rows[source_id] = row
    return rows


def prediction_path_from_summary(summary_path: Path, interface: str) -> Path:
    data = load_json(summary_path)
    rows = data.get("rows", [])
    matches = [row for row in rows if row.get("interface_name") == interface]
    if not matches and len(rows) == 1:
        matches = rows
    if not matches:
        raise ValueError(f"no row for {interface} in {summary_path}")
    return Path(matches[0]["eval_summary_path"]).with_name("predictions.jsonl")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def text_contains(haystack: str, needle: str) -> bool:
    hay = normalize_text(haystack)
    ned = normalize_text(needle)
    return bool(ned) and (ned in hay or hay in ned)


def get_metric(row: dict[str, Any], name: str) -> float:
    return float(row.get("metrics", {}).get(name, 0.0))


def classify_prediction(raw_prediction: str, answer: str, metadata: dict[str, Any]) -> str:
    prediction = normalize_text(raw_prediction)
    if not prediction:
        return "empty prediction"
    if text_contains(raw_prediction, answer):
        return "partial or near-gold answer"

    for item in metadata.get("question_decomposition", []) or []:
        intermediate = str(item.get("answer", ""))
        if intermediate and text_contains(raw_prediction, intermediate):
            return "intermediate-hop answer"

    for title in metadata.get("support_titles", []) or []:
        if title and text_contains(raw_prediction, str(title)):
            return "support-entity answer"

    for title in metadata.get("original_titles", []) or []:
        if title and text_contains(raw_prediction, str(title)):
            return "distractor-entity answer"

    if prediction in {"unknown", "not enough information", "cannot determine", "none"}:
        return "abstention-like answer"
    return "other wrong answer"


def support_layout_tag(metadata: dict[str, Any]) -> str:
    positions = metadata.get("support_positions_original") or metadata.get("support_indices") or []
    doc_count = int(metadata.get("doc_count") or len(metadata.get("original_titles", [])) or 0)
    if not positions or doc_count <= 0:
        return "unknown layout"
    positions = sorted(int(pos) for pos in positions)
    first = positions[0]
    last = positions[-1]
    gap = last - first if len(positions) > 1 else 0
    if first >= max(1, doc_count // 2):
        return "all support late"
    if last >= max(1, int(doc_count * 0.75)):
        return "some support very late"
    if gap >= max(3, doc_count // 3):
        return "support split by distractors"
    if first == 0:
        return "support already starts early"
    return "support mid-context"


def primary_error_tag(raw_prediction: str, answer: str, metadata: dict[str, Any]) -> str:
    pred_tag = classify_prediction(raw_prediction, answer, metadata)
    layout = support_layout_tag(metadata)
    if pred_tag in {"intermediate-hop answer", "support-entity answer", "distractor-entity answer"}:
        return pred_tag
    if layout in {"all support late", "some support very late", "support split by distractors"}:
        return layout
    raw_tokens = int(metadata.get("raw_context_tokens") or 0)
    doc_count = int(metadata.get("doc_count") or 0)
    if raw_tokens >= 1200 or doc_count >= 15:
        return "long distractor-heavy context"
    return pred_tag


def primary_group(tag: str) -> str:
    if tag in {"all support late", "some support very late", "support split by distractors", "long distractor-heavy context"}:
        return "layout bottleneck"
    if tag in {"intermediate-hop answer", "support-entity answer", "distractor-entity answer"}:
        return "entity or hop confusion"
    if tag in {"partial or near-gold answer"}:
        return "answer form near miss"
    return "other wrong answer"


def compact_titles(metadata: dict[str, Any], limit: int = 8) -> str:
    titles = [str(title) for title in metadata.get("original_titles", [])]
    if len(titles) > limit:
        titles = titles[:limit] + ["..."]
    return "; ".join(titles)


def analyze_dataset(spec: DatasetSpec, raw_fail_threshold: float, support_success_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_pred_path = prediction_path_from_summary(RANK_SWEEP / spec.raw_summary, "raw_context")
    support_pred_path = prediction_path_from_summary(RANK_SWEEP / spec.support_summary, "raw_support_first")
    raw_predictions = load_jsonl_by_source_id(raw_pred_path)
    support_predictions = load_jsonl_by_source_id(support_pred_path)
    support_rows = load_jsonl_by_source_id(spec.data_dir / "validation__raw_support_first.jsonl")
    paired_ids = sorted(set(raw_predictions) & set(support_predictions))

    raw_f1_values = [get_metric(raw_predictions[source_id], "f1") for source_id in paired_ids]
    support_f1_values = [get_metric(support_predictions[source_id], "f1") for source_id in paired_ids]
    raw_context_tokens = [
        int((support_rows.get(source_id, {}).get("metadata", {}) or {}).get("raw_context_tokens") or 0)
        for source_id in paired_ids
    ]

    rows: list[dict[str, Any]] = []
    for source_id in paired_ids:
        raw_row = raw_predictions[source_id]
        support_row = support_predictions[source_id]
        metadata = (support_rows.get(source_id) or raw_row).get("metadata", {})
        raw_f1 = get_metric(raw_row, "f1")
        support_f1 = get_metric(support_row, "f1")
        if raw_f1 > raw_fail_threshold or support_f1 < support_success_threshold:
            continue
        prediction_tag = classify_prediction(raw_row.get("prediction", ""), raw_row.get("answer", ""), metadata)
        layout = support_layout_tag(metadata)
        primary = primary_error_tag(raw_row.get("prediction", ""), raw_row.get("answer", ""), metadata)
        rows.append(
            {
                "dataset": spec.key,
                "dataset_label": spec.label,
                "source_id": source_id,
                "question": raw_row.get("question", ""),
                "answer": raw_row.get("answer", ""),
                "raw_prediction": raw_row.get("prediction", ""),
                "support_first_prediction": support_row.get("prediction", ""),
                "raw_f1": round(raw_f1, 4),
                "support_first_f1": round(support_f1, 4),
                "delta_f1": round(support_f1 - raw_f1, 4),
                "primary_tag": primary,
                "primary_group": primary_group(primary),
                "prediction_tag": prediction_tag,
                "layout_tag": layout,
                "support_positions_original": json.dumps(metadata.get("support_positions_original") or metadata.get("support_indices") or []),
                "doc_count": int(metadata.get("doc_count") or 0),
                "raw_context_tokens": int(metadata.get("raw_context_tokens") or 0),
                "support_titles": "; ".join(str(title) for title in metadata.get("support_titles", [])),
                "original_titles_prefix": compact_titles(metadata),
            }
        )

    summary = {
        "dataset": spec.key,
        "dataset_label": spec.label,
        "n_paired": len(paired_ids),
        "raw_mean_f1": round(sum(raw_f1_values) / len(raw_f1_values), 4),
        "support_first_mean_f1": round(sum(support_f1_values) / len(support_f1_values), 4),
        "median_raw_context_tokens": round(median(raw_context_tokens), 1),
        "win_case_count": len(rows),
        "win_case_rate": round(len(rows) / len(paired_ids), 4),
        "raw_fail_threshold": raw_fail_threshold,
        "support_success_threshold": support_success_threshold,
    }
    rows.sort(key=lambda row: (-float(row["delta_f1"]), row["dataset"], row["source_id"]))
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown_summary(path: Path, summaries: list[dict[str, Any]], all_rows: list[dict[str, Any]], examples_per_dataset: int) -> None:
    lines = [
        "# Support-First Error Taxonomy",
        "",
        "Cases are paired by validation `source_id`. A case is counted when",
        "`raw_context` has F1 at or below the raw-fail threshold and",
        "`raw_support_first` has F1 at or above the support-success threshold.",
        "Tags are deterministic diagnostics for sampling and writing; they are",
        "not a substitute for final human annotation.",
        "",
        "## Summary",
        "",
        "|Dataset|Paired n|Raw F1|Support-first F1|Win cases|Win rate|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "|"
            + "|".join(
                [
                    summary["dataset_label"],
                    str(summary["n_paired"]),
                    f"{summary['raw_mean_f1']:.3f}",
                    f"{summary['support_first_mean_f1']:.3f}",
                    str(summary["win_case_count"]),
                    f"{summary['win_case_rate']:.3f}",
                ]
            )
            + "|"
        )

    lines.extend(["", "## Primary Tags", ""])
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for row in all_rows:
        by_dataset[row["dataset_label"]][row["primary_tag"]] += 1
        by_group[row["dataset_label"]][row["primary_group"]] += 1
    lines.extend(["### Grouped Tags", "", "|Dataset|Layout bottleneck|Entity or hop confusion|Answer form near miss|Other wrong answer|", "|---|---:|---:|---:|---:|"])
    for dataset in sorted(by_group):
        counter = by_group[dataset]
        lines.append(
            f"|{dataset}|{counter['layout bottleneck']}|{counter['entity or hop confusion']}|"
            f"{counter['answer form near miss']}|{counter['other wrong answer']}|"
        )
    lines.append("")
    lines.append("### Fine Tags")
    lines.append("")
    for dataset in sorted(by_dataset):
        lines.extend([f"### {dataset}", "", "|Tag|Count|", "|---|---:|"])
        for tag, count in by_dataset[dataset].most_common():
            lines.append(f"|{tag}|{count}|")
        lines.append("")

    lines.extend(["## Review Examples", ""])
    for dataset in sorted({row["dataset_label"] for row in all_rows}):
        lines.append(f"### {dataset}")
        lines.append("")
        dataset_rows = [row for row in all_rows if row["dataset_label"] == dataset][:examples_per_dataset]
        for index, row in enumerate(dataset_rows, start=1):
            lines.extend(
                [
                    f"{index}. **{row['primary_tag']}** (`{row['source_id']}`)",
                    f"   - Q: {row['question']}",
                    f"   - Gold: {row['answer']}",
                    f"   - Raw: {row['raw_prediction']} (F1={row['raw_f1']})",
                    f"   - Support-first: {row['support_first_prediction']} (F1={row['support_first_f1']})",
                    f"   - Support positions: {row['support_positions_original']}; titles: {row['support_titles']}",
                    "",
                ]
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-fail-threshold", type=float, default=0.2)
    parser.add_argument("--support-success-threshold", type=float, default=0.8)
    parser.add_argument("--examples-per-dataset", type=int, default=8)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in DATASETS:
        rows, summary = analyze_dataset(
            spec,
            raw_fail_threshold=args.raw_fail_threshold,
            support_success_threshold=args.support_success_threshold,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    all_rows.sort(key=lambda row: (row["dataset"], -float(row["delta_f1"]), row["source_id"]))
    csv_path = args.out_dir / "ipm_support_first_error_taxonomy_cases.csv"
    jsonl_path = args.out_dir / "ipm_support_first_error_taxonomy_cases.jsonl"
    md_path = args.out_dir / "ipm_support_first_error_taxonomy.md"
    write_csv(csv_path, all_rows)
    write_jsonl(jsonl_path, all_rows)
    write_markdown_summary(md_path, summaries, all_rows, args.examples_per_dataset)
    print(f"wrote {csv_path}")
    print(f"wrote {jsonl_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
