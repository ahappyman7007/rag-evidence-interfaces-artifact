#!/usr/bin/env python3
"""Compute support recall for retrieval-ordered evidence interfaces.

This analysis separates retrieval failure from reader/interface failure.  It
uses the diagnostic metadata already stored in the interface JSONL files and
does not run any model inference.
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
class RetrievalSpec:
    dataset: str
    dataset_label: str
    method: str
    method_label: str
    order_file: Path
    position_key: str


SPECS = [
    RetrievalSpec(
        "hotpotqa",
        "HotpotQA",
        "bm25",
        "BM25",
        ROOT / "data" / "hotpotqa_interfaces" / "pilot" / "validation__raw_bm25_order.jsonl",
        "support_positions_bm25",
    ),
    RetrievalSpec(
        "hotpotqa",
        "HotpotQA",
        "crossenc",
        "Cross-encoder",
        ROOT / "data" / "hotpotqa_interfaces" / "pilot" / "validation__raw_crossenc_order.jsonl",
        "support_positions_crossenc",
    ),
    RetrievalSpec(
        "hotpotqa",
        "HotpotQA",
        "ftcrossenc",
        "Fine-tuned cross-encoder",
        ROOT / "data" / "hotpotqa_interfaces" / "pilot" / "validation__raw_ftcrossenc_order.jsonl",
        "support_positions_crossenc",
    ),
    RetrievalSpec(
        "2wiki",
        "2WikiMultiHopQA",
        "bm25",
        "BM25",
        ROOT / "data" / "2wiki_interfaces" / "pilot" / "validation__raw_bm25_order.jsonl",
        "support_positions_bm25",
    ),
    RetrievalSpec(
        "2wiki",
        "2WikiMultiHopQA",
        "crossenc",
        "Cross-encoder",
        ROOT / "data" / "2wiki_interfaces" / "pilot" / "validation__raw_crossenc_order.jsonl",
        "support_positions_crossenc",
    ),
    RetrievalSpec(
        "musique",
        "MuSiQue",
        "bm25",
        "BM25",
        ROOT / "data" / "musique_interfaces" / "pilot" / "validation__raw_bm25_order.jsonl",
        "support_positions_bm25",
    ),
    RetrievalSpec(
        "musique",
        "MuSiQue",
        "crossenc",
        "Cross-encoder",
        ROOT / "data" / "musique_interfaces" / "pilot" / "validation__raw_crossenc_order.jsonl",
        "support_positions_crossenc",
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


def support_count(meta: dict[str, Any]) -> int:
    if meta.get("support_doc_count") is not None:
        return int(meta["support_doc_count"])
    titles = meta.get("support_titles") or []
    if titles:
        return len({str(title).strip().lower() for title in titles})
    positions = meta.get("support_positions_original") or []
    if positions:
        return len(set(int(pos) for pos in positions))
    return int(meta.get("supporting_fact_count", 0))


def summarize_spec(spec: RetrievalSpec, ks: list[int]) -> list[dict[str, Any]]:
    rows = read_jsonl(spec.order_file)
    if not rows:
        raise ValueError(f"empty file: {spec.order_file}")

    per_k: list[dict[str, Any]] = []
    for k in ks:
        any_hits: list[float] = []
        all_hits: list[float] = []
        support_fracs: list[float] = []
        support_counts: list[int] = []
        effective_ks: list[int] = []
        latest_ranks: list[int] = []

        for row in rows:
            meta = row.get("metadata", {})
            positions = [int(pos) for pos in meta.get(spec.position_key, [])]
            count = support_count(meta)
            doc_count = int(meta.get("doc_count", len(meta.get("ordered_titles", [])) or k))
            effective_k = min(k, doc_count)
            in_top_k = [pos for pos in positions if pos < effective_k]
            matched = len(set(in_top_k))

            any_hits.append(1.0 if matched > 0 else 0.0)
            all_hits.append(1.0 if count > 0 and matched >= count else 0.0)
            support_fracs.append(float(matched / count) if count else 0.0)
            support_counts.append(matched)
            effective_ks.append(effective_k)
            if positions:
                latest_ranks.append(max(positions) + 1)

        per_k.append(
            {
                "dataset": spec.dataset,
                "dataset_label": spec.dataset_label,
                "method": spec.method,
                "method_label": spec.method_label,
                "k": k,
                "examples": len(rows),
                "avg_effective_k": round(mean(effective_ks), 3),
                "any_support_recall": round(mean(any_hits), 6),
                "all_support_recall": round(mean(all_hits), 6),
                "mean_support_fraction": round(mean(support_fracs), 6),
                "avg_support_docs_in_topk": round(mean(support_counts), 6),
                "mean_latest_support_rank": round(mean(latest_ranks), 6) if latest_ranks else "",
                "source_file": str(spec.order_file.relative_to(ROOT)),
            }
        )
    return per_k


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    wanted_ks = [5, 10, 20]
    compact = [row for row in rows if int(row["k"]) in wanted_ks]
    headers = [
        "Dataset",
        "Method",
        "k",
        "Any support",
        "All support",
        "Mean support frac.",
        "Mean latest support rank",
    ]
    lines = [
        "# Retrieval Support Recall",
        "",
        "Computed from validation interface metadata. `All support` is the main",
        "multi-hop retrieval diagnostic: the fraction of examples where every",
        "support document appears within the top-k window.",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in compact:
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["dataset_label"]),
                    str(row["method_label"]),
                    str(row["k"]),
                    f"{float(row['any_support_recall']):.3f}",
                    f"{float(row['all_support_recall']):.3f}",
                    f"{float(row['mean_support_fraction']):.3f}",
                    f"{float(row['mean_latest_support_rank']):.2f}",
                ]
            )
            + "|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_quality_cost_rows(recall_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join the main Qwen14B result table with support recall diagnostics."""
    main_path = DEFAULT_OUT_DIR / "ipm_qwen14b_n4000_interface_hierarchy.csv"
    main_rows = read_csv(main_path)
    by_dataset_interface = {
        (row["dataset"], row["interface_name"]): row for row in main_rows
    }
    recall_by_dataset_method_k = {
        (row["dataset"], row["method"], int(row["k"])): row for row in recall_rows
    }
    main_top5 = {
        "hotpotqa": ("raw_ftcrossenc_top5docs", "ftcrossenc"),
        "2wiki": ("raw_crossenc_top5docs", "crossenc"),
        "musique": ("raw_crossenc_top5docs", "crossenc"),
    }

    joined: list[dict[str, Any]] = []
    for dataset, (interface_name, method) in main_top5.items():
        raw = by_dataset_interface[(dataset, "raw_context")]
        top5 = by_dataset_interface[(dataset, interface_name)]
        support_first = by_dataset_interface[(dataset, "raw_support_first")]
        recall = recall_by_dataset_method_k[(dataset, method, 5)]
        raw_f1 = float(raw["f1"])
        top5_f1 = float(top5["f1"])
        support_first_f1 = float(support_first["f1"])
        raw_tokens = float(raw["avg_prompt_tokens"])
        top5_tokens = float(top5["avg_prompt_tokens"])
        joined.append(
            {
                "dataset": dataset,
                "dataset_label": raw["dataset_label"],
                "top5_interface": interface_name,
                "top5_method": recall["method_label"],
                "all_support_recall_at5": recall["all_support_recall"],
                "mean_support_fraction_at5": recall["mean_support_fraction"],
                "raw_f1": round(raw_f1, 6),
                "top5_f1": round(top5_f1, 6),
                "support_first_f1": round(support_first_f1, 6),
                "top5_minus_raw_f1": round(top5_f1 - raw_f1, 6),
                "support_first_minus_raw_f1": round(support_first_f1 - raw_f1, 6),
                "raw_avg_prompt_tokens": round(raw_tokens, 2),
                "top5_avg_prompt_tokens": round(top5_tokens, 2),
                "token_reduction_vs_raw": round(1.0 - (top5_tokens / raw_tokens), 6),
            }
        )
    return joined


def write_quality_cost_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "Dataset",
        "Top-5 method",
        "All support@5",
        "Raw F1",
        "Top-5 F1",
        "Top-5 - raw",
        "Raw tokens",
        "Top-5 tokens",
        "Token reduction",
    ]
    lines = [
        "# Retrieval Recall, Quality, and Cost",
        "",
        "This table joins validation support recall with the current Qwen14B",
        "n=4000 main result table. It is intended for the retrieval",
        "quality/cost section.",
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
                    str(row["top5_method"]),
                    f"{float(row['all_support_recall_at5']):.3f}",
                    f"{float(row['raw_f1']):.3f}",
                    f"{float(row['top5_f1']):.3f}",
                    f"{float(row['top5_minus_raw_f1']):+.3f}",
                    f"{float(row['raw_avg_prompt_tokens']):.0f}",
                    f"{float(row['top5_avg_prompt_tokens']):.0f}",
                    f"{100 * float(row['token_reduction_vs_raw']):.1f}%",
                ]
            )
            + "|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for spec in SPECS:
        if not spec.order_file.exists():
            raise FileNotFoundError(spec.order_file)
        rows.extend(summarize_spec(spec, args.ks))

    csv_path = args.out_dir / "ipm_retrieval_support_recall.csv"
    md_path = args.out_dir / "ipm_retrieval_support_recall.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    quality_cost_rows = build_quality_cost_rows(rows)
    quality_cost_csv = args.out_dir / "ipm_retrieval_recall_quality_cost.csv"
    quality_cost_md = args.out_dir / "ipm_retrieval_recall_quality_cost.md"
    write_csv(quality_cost_csv, quality_cost_rows)
    write_quality_cost_markdown(quality_cost_md, quality_cost_rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {quality_cost_csv}")
    print(f"wrote {quality_cost_md}")


if __name__ == "__main__":
    main()
