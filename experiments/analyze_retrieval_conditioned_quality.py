#!/usr/bin/env python3
"""Condition top-k reader quality on whether the retrieved window keeps support.

This is a paper-facing diagnostic for the IPM manuscript. It joins Qwen14B
raw-context and top-5 prediction files by validation example and splits the
top-5 results by complete-support coverage. The goal is to separate two
failure modes: a top-k window can be worse because the reader cannot use the
shown evidence, or because the window simply omits part of the support chain.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "experiments" / "outputs" / "consolidated" / "tables"


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    dataset_label: str
    top5_method: str
    raw_predictions: Path
    top5_predictions: Path


SPECS = [
    DatasetSpec(
        dataset="hotpotqa",
        dataset_label="HotpotQA",
        top5_method="Fine-tuned cross-encoder",
        raw_predictions=ROOT
        / "experiments"
        / "outputs"
        / "smoke_test_hotpotqa_qwen14b"
        / "raw_context_lora_r8_n4000_eval300"
        / "predictions.jsonl",
        top5_predictions=ROOT
        / "experiments"
        / "outputs"
        / "smoke_test_hotpotqa_qwen14b"
        / "raw_ftcrossenc_top5docs_lora_r8_n4000_eval300"
        / "predictions.jsonl",
    ),
    DatasetSpec(
        dataset="2wiki",
        dataset_label="2WikiMultiHopQA",
        top5_method="Cross-encoder",
        raw_predictions=ROOT
        / "experiments"
        / "outputs"
        / "smoke_test_2wiki_qwen14b"
        / "raw_context_lora_r8_n4000_eval300"
        / "predictions.jsonl",
        top5_predictions=ROOT
        / "experiments"
        / "outputs"
        / "smoke_test_2wiki_qwen14b"
        / "raw_crossenc_top5docs_lora_r8_n4000_eval300"
        / "predictions.jsonl",
    ),
    DatasetSpec(
        dataset="musique",
        dataset_label="MuSiQue",
        top5_method="Cross-encoder",
        raw_predictions=ROOT
        / "experiments"
        / "outputs"
        / "smoke_test_musique_qwen14b"
        / "raw_context_lora_r8_n4000_eval300"
        / "predictions.jsonl",
        top5_predictions=ROOT
        / "experiments"
        / "outputs"
        / "smoke_test_musique_qwen14b"
        / "raw_crossenc_top5docs_lora_r8_n4000_eval300"
        / "predictions.jsonl",
    ),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prediction_f1(row: dict[str, Any]) -> float:
    return float(row.get("metrics", {}).get("f1", 0.0))


def support_doc_count(meta: dict[str, Any]) -> int:
    if meta.get("support_doc_count") is not None:
        return int(meta["support_doc_count"])
    titles = meta.get("support_titles") or []
    if titles:
        return len({str(title).strip().lower() for title in titles})
    if meta.get("supporting_fact_count") is not None:
        return int(meta["supporting_fact_count"])
    return 0


def matched_support_count(meta: dict[str, Any]) -> int:
    if meta.get("matched_support_doc_count_window") is not None:
        return int(meta["matched_support_doc_count_window"])
    if meta.get("support_doc_count_in_window") is not None:
        return int(meta["support_doc_count_in_window"])
    positions = meta.get("support_positions_window") or []
    return len({int(pos) for pos in positions})


def completeness_label(meta: dict[str, Any]) -> tuple[str, float]:
    total = support_doc_count(meta)
    matched = matched_support_count(meta)
    frac = float(matched / total) if total else 0.0
    if total > 0 and matched >= total:
        return "Complete support in top-5", frac
    return "Missing support in top-5", frac


def summarize_spec(spec: DatasetSpec) -> list[dict[str, Any]]:
    raw_rows = read_jsonl(spec.raw_predictions)
    top5_rows = read_jsonl(spec.top5_predictions)
    raw_by_id = {str(row["source_id"]): row for row in raw_rows}

    joined: dict[str, list[dict[str, float]]] = {
        "Complete support in top-5": [],
        "Missing support in top-5": [],
    }

    for top5 in top5_rows:
        source_id = str(top5["source_id"])
        raw = raw_by_id.get(source_id)
        if raw is None:
            raise KeyError(f"missing raw prediction for {source_id} in {spec.dataset}")
        label, frac = completeness_label(top5.get("metadata", {}))
        joined[label].append(
            {
                "raw_f1": prediction_f1(raw),
                "top5_f1": prediction_f1(top5),
                "support_fraction": frac,
            }
        )

    rows: list[dict[str, Any]] = []
    total_examples = sum(len(items) for items in joined.values())
    for label, items in joined.items():
        if not items:
            continue
        raw_f1 = mean(item["raw_f1"] for item in items)
        top5_f1 = mean(item["top5_f1"] for item in items)
        support_fraction = mean(item["support_fraction"] for item in items)
        rows.append(
            {
                "dataset": spec.dataset,
                "dataset_label": spec.dataset_label,
                "top5_method": spec.top5_method,
                "support_status": label,
                "examples": len(items),
                "share": round(len(items) / total_examples, 6),
                "mean_support_fraction": round(support_fraction, 6),
                "raw_f1": round(raw_f1, 6),
                "top5_f1": round(top5_f1, 6),
                "top5_minus_raw_f1": round(top5_f1 - raw_f1, 6),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "Dataset",
        "Support status",
        "n",
        "Share",
        "Mean supp. frac.",
        "Raw F1",
        "Top-5 F1",
        "Top-5 - raw",
    ]
    lines = [
        "# Retrieval-Conditioned Reader Quality",
        "",
        "Rows split Qwen14B n=4000 top-5 predictions by whether the retrieved",
        "window contains every annotated support unit for the same validation",
        "example. Raw F1 is computed on the same subset, so the final column",
        "separates coverage failures from reader behavior when support is present.",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["dataset_label"]),
                    str(row["support_status"]).replace(" in top-5", ""),
                    str(row["examples"]),
                    f"{100 * float(row['share']):.1f}%",
                    f"{float(row['mean_support_fraction']):.3f}",
                    f"{float(row['raw_f1']):.3f}",
                    f"{float(row['top5_f1']):.3f}",
                    f"{float(row['top5_minus_raw_f1']):+.3f}",
                ]
            )
            + "|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        rows.extend(summarize_spec(spec))

    write_csv(out_dir / "ipm_retrieval_conditioned_quality.csv", rows)
    write_markdown(out_dir / "ipm_retrieval_conditioned_quality.md", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
