#!/usr/bin/env python
"""Compare raw-context and support-first HotpotQA prediction errors."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

from inspect_hotpotqa import (
    approx_tokens,
    context_docs,
    extract_gold_sentences,
    flatten_context,
    load_split,
    normalize_text,
    supporting_facts,
)


EXPERIMENTS = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = EXPERIMENTS / "outputs" / "error_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw vs support-first prediction flips.")
    parser.add_argument("--dataset-name", default="hotpotqa/hotpot_qa")
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--support-first-predictions", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def source_id(row: Dict[str, Any]) -> str:
    return str(row.get("source_id") or row.get("metadata", {}).get("source_id") or "")


def support_titles_from_facts(facts: Sequence[tuple[str, int]]) -> List[str]:
    titles: List[str] = []
    seen = set()
    for title, _sent_id in facts:
        if title not in seen:
            titles.append(str(title))
            seen.add(title)
    return titles


def normalized_count(text: str, answer: str) -> int:
    answer_norm = normalize_text(answer)
    text_norm = normalize_text(text)
    if not answer_norm or answer_norm in {"yes", "no", "noanswer"}:
        return 0
    pattern = rf"(?<!\w){re.escape(answer_norm)}(?!\w)"
    return len(re.findall(pattern, text_norm))


def doc_text(doc: Dict[str, Any]) -> str:
    return f"{doc['title']}: {' '.join(doc['sentences'])}"


def evidence_features(example: Dict[str, Any]) -> Dict[str, Any]:
    docs = context_docs(example)
    facts = supporting_facts(example)
    gold_sentences, missing = extract_gold_sentences(docs, facts)
    support_titles = support_titles_from_facts(facts)
    support_set = set(support_titles)
    titles = [str(doc["title"]) for doc in docs]
    support_positions = [idx for idx, title in enumerate(titles) if title in support_set]

    support_docs = [doc for doc in docs if str(doc["title"]) in support_set]
    nonsupport_docs = [doc for doc in docs if str(doc["title"]) not in support_set]
    raw_context = flatten_context(docs)
    support_context = flatten_context(support_docs)
    nonsupport_context = flatten_context(nonsupport_docs)
    answer = str(example.get("answer", "")).strip()

    if support_positions:
        earliest = min(support_positions)
        latest = max(support_positions)
        span = latest - earliest
        mean_pos = mean(support_positions)
        latest_frac = latest / max(len(docs) - 1, 1)
        earliest_frac = earliest / max(len(docs) - 1, 1)
    else:
        earliest = latest = span = -1
        mean_pos = -1.0
        latest_frac = earliest_frac = -1.0

    raw_count = normalized_count(raw_context, answer)
    support_count = normalized_count(support_context, answer)
    nonsupport_count = normalized_count(nonsupport_context, answer)

    return {
        "source_id": str(example.get("id", "")),
        "type": str(example.get("type", "")),
        "level": str(example.get("level", "")),
        "answer_type": "yes_no" if normalize_text(answer) in {"yes", "no"} else "span",
        "doc_count": len(docs),
        "sentence_count": sum(len(doc["sentences"]) for doc in docs),
        "supporting_fact_count": len(facts),
        "matched_gold_sentence_count": len(gold_sentences),
        "missing_supporting_fact_count": len(missing),
        "support_doc_count": len(support_titles),
        "matched_support_doc_count": len(support_positions),
        "support_titles": "|".join(support_titles),
        "support_positions": "|".join(str(pos) for pos in support_positions),
        "earliest_support_index": earliest,
        "latest_support_index": latest,
        "mean_support_index": mean_pos,
        "support_span_width": span,
        "earliest_support_frac": earliest_frac,
        "latest_support_frac": latest_frac,
        "has_support_at_first_doc": int(bool(support_positions and min(support_positions) == 0)),
        "has_support_at_last_doc": int(bool(support_positions and max(support_positions) == len(docs) - 1)),
        "raw_context_tokens": approx_tokens(raw_context),
        "gold_evidence_tokens": approx_tokens(" ".join(gold_sentences)),
        "answer_count_raw_context": raw_count,
        "answer_count_support_docs": support_count,
        "answer_count_nonsupport_docs": nonsupport_count,
        "answer_in_nonsupport_docs": int(nonsupport_count > 0),
        "answer_only_in_nonsupport_docs": int(nonsupport_count > 0 and support_count == 0),
    }


def correctness(row: Dict[str, Any]) -> bool:
    metrics = row.get("metrics", {})
    return bool(metrics.get("exact_match", 0.0) >= 1.0)


def category(raw_row: Dict[str, Any], support_row: Dict[str, Any]) -> str:
    raw_ok = correctness(raw_row)
    support_ok = correctness(support_row)
    if raw_ok and support_ok:
        return "both_correct"
    if raw_ok and not support_ok:
        return "raw_only"
    if not raw_ok and support_ok:
        return "support_first_only"
    return "both_wrong"


def latest_bucket(latest: int) -> str:
    if latest < 0:
        return "missing"
    if latest <= 2:
        return "latest_0_2"
    if latest <= 5:
        return "latest_3_5"
    return "latest_6_9"


def numeric_mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in {None, ""}]
    return mean(values) if values else 0.0


def summarize_group(rows: Sequence[Dict[str, Any]], group_name: str, group_value: str) -> Dict[str, Any]:
    if not rows:
        return {
            "group_name": group_name,
            "group_value": group_value,
            "n": 0,
            "raw_em": 0.0,
            "support_first_em": 0.0,
            "raw_f1": 0.0,
            "support_first_f1": 0.0,
            "avg_latest_support_index": 0.0,
            "avg_support_span_width": 0.0,
            "answer_in_nonsupport_rate": 0.0,
        }
    return {
        "group_name": group_name,
        "group_value": group_value,
        "n": len(rows),
        "raw_em": numeric_mean(rows, "raw_exact_match"),
        "support_first_em": numeric_mean(rows, "support_first_exact_match"),
        "raw_f1": numeric_mean(rows, "raw_f1"),
        "support_first_f1": numeric_mean(rows, "support_first_f1"),
        "avg_latest_support_index": numeric_mean(rows, "latest_support_index"),
        "avg_support_span_width": numeric_mean(rows, "support_span_width"),
        "answer_in_nonsupport_rate": numeric_mean(rows, "answer_in_nonsupport_docs"),
    }


def grouped_rows(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return [summarize_group(items, key, value) for value, items in sorted(grouped.items())]


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = {source_id(row): row for row in read_jsonl(args.raw_predictions)}
    support_rows = {source_id(row): row for row in read_jsonl(args.support_first_predictions)}
    common_ids = sorted(set(raw_rows) & set(support_rows))
    if not common_ids:
        raise RuntimeError("No overlapping source_id values between prediction files")

    ds = load_split(args.dataset_name, args.config, args.split)

    examples: List[Dict[str, Any]] = []
    for sid in common_ids:
        raw_row = raw_rows[sid]
        support_row = support_rows[sid]
        source_index = int(raw_row.get("metadata", {}).get("source_index", -1))
        if source_index < 0:
            raise ValueError(f"Missing source_index for {sid}")
        features = evidence_features(ds[source_index])
        cat = category(raw_row, support_row)
        record = {
            "source_id": sid,
            "category": cat,
            "question": raw_row.get("question", ""),
            "answer": raw_row.get("answer", ""),
            "raw_prediction": raw_row.get("prediction", ""),
            "support_first_prediction": support_row.get("prediction", ""),
            "raw_exact_match": raw_row["metrics"]["exact_match"],
            "support_first_exact_match": support_row["metrics"]["exact_match"],
            "raw_f1": raw_row["metrics"]["f1"],
            "support_first_f1": support_row["metrics"]["f1"],
            "raw_answer_contained": raw_row["metrics"]["answer_contained"],
            "support_first_answer_contained": support_row["metrics"]["answer_contained"],
            "raw_prompt_tokens": raw_row["generation"]["prompt_tokens"],
            "support_first_prompt_tokens": support_row["generation"]["prompt_tokens"],
            "latest_support_bucket": latest_bucket(features["latest_support_index"]),
            **features,
        }
        examples.append(record)

    examples.sort(key=lambda row: (row["category"], row["source_id"]))

    category_counts = Counter(row["category"] for row in examples)
    category_summary = grouped_rows(examples, "category")
    type_summary = grouped_rows(examples, "type")
    latest_bucket_summary = grouped_rows(examples, "latest_support_bucket")

    support_first_only = [row for row in examples if row["category"] == "support_first_only"]
    raw_only = [row for row in examples if row["category"] == "raw_only"]
    summary = {
        "run_name": args.run_name,
        "raw_predictions": str(args.raw_predictions),
        "support_first_predictions": str(args.support_first_predictions),
        "n": len(examples),
        "category_counts": dict(category_counts),
        "category_rates": {key: value / len(examples) for key, value in category_counts.items()},
        "overall": summarize_group(examples, "overall", "all"),
        "support_first_only": summarize_group(support_first_only, "category", "support_first_only"),
        "raw_only": summarize_group(raw_only, "category", "raw_only"),
        "support_first_only_latest_6_9_rate": (
            sum(row["latest_support_bucket"] == "latest_6_9" for row in support_first_only)
            / len(support_first_only)
            if support_first_only
            else 0.0
        ),
        "support_first_only_answer_in_nonsupport_rate": (
            mean(row["answer_in_nonsupport_docs"] for row in support_first_only)
            if support_first_only
            else 0.0
        ),
        "category_summary": category_summary,
        "type_summary": type_summary,
        "latest_bucket_summary": latest_bucket_summary,
    }

    write_csv(output_dir / "examples.csv", examples)
    write_csv(output_dir / "category_summary.csv", category_summary)
    write_csv(output_dir / "type_summary.csv", type_summary)
    write_csv(output_dir / "latest_bucket_summary.csv", latest_bucket_summary)
    write_json(output_dir / "summary.json", summary)

    print(f"wrote {output_dir}")
    print("category,n,raw_EM,support_EM,raw_F1,support_F1,avg_latest,avg_span,answer_nonsupport")
    for row in category_summary:
        print(
            f"{row['group_value']},{row['n']},"
            f"{row['raw_em']:.3f},{row['support_first_em']:.3f},"
            f"{row['raw_f1']:.3f},{row['support_first_f1']:.3f},"
            f"{row['avg_latest_support_index']:.2f},"
            f"{row['avg_support_span_width']:.2f},"
            f"{row['answer_in_nonsupport_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
