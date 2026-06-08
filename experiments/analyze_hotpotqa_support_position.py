#!/usr/bin/env python
"""Analyze HotpotQA raw-context performance by supporting-document position."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

from inspect_hotpotqa import context_docs, load_split, supporting_facts


DEFAULT_PREDICTIONS = (
    Path(__file__).resolve().parent
    / "outputs"
    / "smoke_test"
    / "raw_context_lora_r8_n500_eval300"
    / "predictions.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "position_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw-context metrics by support-document position.")
    parser.add_argument("--dataset-name", default="hotpotqa/hotpot_qa")
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def support_titles_from_facts(facts: Sequence[tuple[str, int]]) -> List[str]:
    titles: List[str] = []
    seen = set()
    for title, _sent_id in facts:
        if title not in seen:
            titles.append(str(title))
            seen.add(title)
    return titles


def position_info(example: Dict[str, Any]) -> Dict[str, Any]:
    docs = context_docs(example)
    facts = supporting_facts(example)
    support_titles = support_titles_from_facts(facts)
    support_set = set(support_titles)
    titles = [str(doc["title"]) for doc in docs]
    positions = [idx for idx, title in enumerate(titles) if title in support_set]
    doc_count = len(docs)
    if positions:
        earliest = min(positions)
        latest = max(positions)
        mean_pos = mean(positions)
        latest_frac = latest / max(doc_count - 1, 1)
        earliest_frac = earliest / max(doc_count - 1, 1)
    else:
        earliest = latest = -1
        mean_pos = -1.0
        earliest_frac = latest_frac = -1.0
    return {
        "doc_count": doc_count,
        "support_titles": support_titles,
        "original_titles": titles,
        "support_positions": positions,
        "earliest_support_index": earliest,
        "latest_support_index": latest,
        "mean_support_index": mean_pos,
        "earliest_support_frac": earliest_frac,
        "latest_support_frac": latest_frac,
        "support_doc_count": len(support_titles),
        "matched_support_doc_count": len(positions),
    }


def bucket_by_latest(latest: int, doc_count: int) -> str:
    if latest < 0:
        return "missing"
    if latest <= 2:
        return "latest_0_2"
    if latest <= 5:
        return "latest_3_5"
    return f"latest_6_{max(doc_count - 1, latest)}"


def bucket_by_earliest(earliest: int, doc_count: int) -> str:
    if earliest < 0:
        return "missing"
    if earliest <= 2:
        return "earliest_0_2"
    if earliest <= 5:
        return "earliest_3_5"
    return f"earliest_6_{max(doc_count - 1, earliest)}"


def bucket_by_latest_tertile(latest_frac: float) -> str:
    if latest_frac < 0:
        return "missing"
    if latest_frac < 1 / 3:
        return "latest_frac_early"
    if latest_frac < 2 / 3:
        return "latest_frac_middle"
    return "latest_frac_late"


def bucket_by_any_first(positions: Sequence[int]) -> str:
    if not positions:
        return "missing"
    return "has_support_at_0" if min(positions) == 0 else "no_support_at_0"


def summarize_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "exact_match": 0.0,
            "f1": 0.0,
            "answer_contained": 0.0,
            "avg_prompt_tokens": 0.0,
            "truncated_count": 0,
            "avg_earliest_support_index": 0.0,
            "avg_latest_support_index": 0.0,
            "avg_latest_support_frac": 0.0,
        }
    return {
        "n": len(rows),
        "exact_match": mean(row["metrics"]["exact_match"] for row in rows),
        "f1": mean(row["metrics"]["f1"] for row in rows),
        "answer_contained": mean(row["metrics"]["answer_contained"] for row in rows),
        "avg_prompt_tokens": mean(row["generation"]["prompt_tokens"] for row in rows),
        "truncated_count": sum(row["generation"]["truncated"] for row in rows),
        "avg_earliest_support_index": mean(row["position"]["earliest_support_index"] for row in rows),
        "avg_latest_support_index": mean(row["position"]["latest_support_index"] for row in rows),
        "avg_latest_support_frac": mean(row["position"]["latest_support_frac"] for row in rows),
    }


def grouped_summary(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["buckets"][key]].append(row)
    return [
        {"bucket_type": key, "bucket": bucket, **summarize_group(items)}
        for bucket, items in sorted(grouped.items())
    ]


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_name = args.run_name or args.predictions.parent.name
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_jsonl(args.predictions)
    ds = load_split(args.dataset_name, args.config, args.split)

    enriched: List[Dict[str, Any]] = []
    for row in predictions:
        source_index = int(row.get("metadata", {}).get("source_index", -1))
        if source_index < 0:
            raise ValueError(f"Prediction row is missing metadata.source_index: {row.get('id')}")
        info = position_info(ds[source_index])
        row = dict(row)
        row["position"] = info
        row["buckets"] = {
            "latest_index": bucket_by_latest(info["latest_support_index"], info["doc_count"]),
            "earliest_index": bucket_by_earliest(info["earliest_support_index"], info["doc_count"]),
            "latest_frac": bucket_by_latest_tertile(info["latest_support_frac"]),
            "support_at_first_doc": bucket_by_any_first(info["support_positions"]),
        }
        enriched.append(row)

    bucket_rows: List[Dict[str, Any]] = []
    for key in ["latest_index", "earliest_index", "latest_frac", "support_at_first_doc"]:
        bucket_rows.extend(grouped_summary(enriched, key))

    overall = {
        "run_name": run_name,
        "predictions": str(args.predictions),
        "n": len(enriched),
        "overall": summarize_group(enriched),
        "bucket_rows": bucket_rows,
    }

    write_json(output_dir / "summary.json", overall)
    write_jsonl = output_dir / "enriched_predictions.jsonl"
    with write_jsonl.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(output_dir / "bucket_metrics.csv", bucket_rows)

    print(f"wrote {output_dir}")
    print("bucket_type,bucket,n,EM,F1,contains")
    for row in bucket_rows:
        print(
            f"{row['bucket_type']},{row['bucket']},{row['n']},"
            f"{row['exact_match']:.3f},{row['f1']:.3f},{row['answer_contained']:.3f}"
        )


if __name__ == "__main__":
    main()
