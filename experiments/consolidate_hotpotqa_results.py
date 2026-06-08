#!/usr/bin/env python
"""Consolidate HotpotQA pilot results into stable tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
OUTPUTS = EXPERIMENTS / "outputs"
DEFAULT_OUT = OUTPUTS / "consolidated"

RANK_SWEEP = OUTPUTS / "rank_sweep"
SUBSPACE = OUTPUTS / "subspace_sweep"

INTERFACE_LABELS = {
    "raw_context": "Raw context",
    "raw_support_first": "Support first",
    "raw_support_first_2docs": "Support first 2 docs",
    "raw_support_first_3docs": "Support first 3 docs",
    "raw_support_first_5docs": "Support first 5 docs",
    "raw_bm25_order": "BM25 order",
    "raw_bm25_top3docs": "BM25 top 3 docs",
    "raw_bm25_top5docs": "BM25 top 5 docs",
    "raw_embed_order": "Embed order",
    "raw_embed_top3docs": "Embed top 3 docs",
    "raw_embed_top5docs": "Embed top 5 docs",
    "raw_crossenc_order": "Cross-enc order",
    "raw_crossenc_top3docs": "Cross-enc top 3 docs",
    "raw_crossenc_top5docs": "Cross-enc top 5 docs",
    "raw_suptitle_order": "Sup-title order",
    "raw_suptitle_top3docs": "Sup-title top 3 docs",
    "raw_suptitle_top5docs": "Sup-title top 5 docs",
    "raw_ftcrossenc_order": "FT cross-enc order",
    "raw_ftcrossenc_top3docs": "FT cross-enc top 3 docs",
    "raw_ftcrossenc_top5docs": "FT cross-enc top 5 docs",
    "raw_support_middle": "Support middle",
    "raw_support_shuffled": "Support shuffled",
    "raw_support_last": "Support last",
    "gold_supporting_sentences": "Gold evidence",
    "gold_plus_1_distractor": "Gold + 1 distractor",
    "gold_plus_3_distractors": "Gold + 3 distractors",
}

INTERFACE_ORDER = [
    "raw_context",
    "raw_support_shuffled",
    "raw_support_middle",
    "raw_support_last",
    "raw_support_first",
    "raw_support_first_2docs",
    "raw_support_first_3docs",
    "raw_support_first_5docs",
    "raw_bm25_order",
    "raw_bm25_top3docs",
    "raw_bm25_top5docs",
    "raw_embed_order",
    "raw_embed_top3docs",
    "raw_embed_top5docs",
    "raw_crossenc_order",
    "raw_crossenc_top3docs",
    "raw_crossenc_top5docs",
    "raw_suptitle_order",
    "raw_suptitle_top3docs",
    "raw_suptitle_top5docs",
    "raw_ftcrossenc_order",
    "raw_ftcrossenc_top3docs",
    "raw_ftcrossenc_top5docs",
    "gold_supporting_sentences",
    "gold_plus_1_distractor",
    "gold_plus_3_distractors",
]

COLORS = {
    "raw_context": "#3b82f6",
    "raw_support_first": "#10b981",
    "raw_support_first_2docs": "#0f766e",
    "raw_support_first_3docs": "#14b8a6",
    "raw_support_first_5docs": "#2dd4bf",
    "raw_bm25_order": "#dc2626",
    "raw_bm25_top3docs": "#f97316",
    "raw_bm25_top5docs": "#fb923c",
    "raw_embed_order": "#7c3aed",
    "raw_embed_top3docs": "#a855f7",
    "raw_embed_top5docs": "#c084fc",
    "raw_crossenc_order": "#0891b2",
    "raw_crossenc_top3docs": "#06b6d4",
    "raw_crossenc_top5docs": "#67e8f9",
    "raw_suptitle_order": "#65a30d",
    "raw_suptitle_top3docs": "#84cc16",
    "raw_suptitle_top5docs": "#bef264",
    "raw_ftcrossenc_order": "#0f766e",
    "raw_ftcrossenc_top3docs": "#14b8a6",
    "raw_ftcrossenc_top5docs": "#5eead4",
    "raw_support_middle": "#8b5cf6",
    "raw_support_shuffled": "#64748b",
    "raw_support_last": "#f97316",
    "gold_supporting_sentences": "#111827",
    "gold_plus_1_distractor": "#f59e0b",
    "gold_plus_3_distractors": "#ef4444",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate HotpotQA experiment outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def rows_from_summary(path: Path, model_label: str | None = None) -> List[Dict[str, Any]]:
    data = load_json(path)
    rows = []
    for row in data.get("rows", []):
        normalized = dict(row)
        normalized["model_id"] = data.get("model_id")
        normalized["model_label"] = model_label or model_short_name(data.get("model_id", ""))
        normalized["source_summary"] = str(path.relative_to(OUTPUTS))
        rows.append(normalized)
    return rows


def model_short_name(model_id: str) -> str:
    if "Llama-3.2-3B" in model_id:
        return "Llama-3.2-3B"
    if "Llama-3.2-1B" in model_id:
        return "Llama-3.2-1B"
    return model_id.rsplit("/", 1)[-1] if model_id else ""


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def round_row(row: Dict[str, Any]) -> Dict[str, Any]:
    rounded = dict(row)
    for key in ["final_eval_loss", "exact_match", "f1", "answer_contained", "em_mean", "f1_mean", "em_sd", "f1_sd"]:
        if key in rounded and isinstance(rounded[key], float):
            rounded[key] = round(rounded[key], 6)
    return rounded


def collect_full_lora_scaling_1b() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    paths = [
        RANK_SWEEP / "hotpotqa_lora_rank_sweep_summary.json",
        RANK_SWEEP / "hotpotqa_evidence_quality_lora_r8_summary.json",
        RANK_SWEEP / "hotpotqa_data_scaling_n1000_summary.json",
        RANK_SWEEP / "hotpotqa_data_scaling_n2000_summary.json",
        RANK_SWEEP / "hotpotqa_raw_context_scaling_n4000_summary.json",
        RANK_SWEEP / "hotpotqa_support_first_scaling_n1000_summary.json",
        RANK_SWEEP / "hotpotqa_support_first_scaling_n2000_summary.json",
        RANK_SWEEP / "hotpotqa_support_first_scaling_n4000_summary.json",
        RANK_SWEEP / "hotpotqa_gold_scaling_n4000_summary.json",
    ]
    for path in paths:
        rows.extend(rows_from_summary(path, model_label="Llama-3.2-1B"))

    wanted = {"raw_context", "raw_support_first", "gold_supporting_sentences"}
    dedup: Dict[tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        if row.get("rank") != 8:
            continue
        if row["interface_name"] not in wanted:
            continue
        key = (row["interface_name"], int(row["train_size"]))
        # Later explicit scaling summaries should override broader sweeps.
        dedup[key] = row

    out = []
    for interface_name in ["raw_context", "raw_support_first", "gold_supporting_sentences"]:
        for train_size in [500, 1000, 2000, 4000]:
            row = dedup.get((interface_name, train_size))
            if not row:
                continue
            out.append(
                round_row(
                    {
                        "model_label": "Llama-3.2-1B",
                        "interface_name": interface_name,
                        "interface_label": INTERFACE_LABELS[interface_name],
                        "train_size": train_size,
                        "updates": row["updates"],
                        "final_eval_loss": row["final_eval_loss"],
                        "exact_match": row["exact_match"],
                        "f1": row["f1"],
                        "answer_contained": row["answer_contained"],
                        "avg_prompt_tokens": row["avg_prompt_tokens"],
                        "truncated_count": row["truncated_count"],
                        "source_summary": row["source_summary"],
                    }
                )
            )
    return out


def collect_larger_model_n500(scaling_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in scaling_rows:
        if row["train_size"] == 500:
            rows.append(dict(row))

    for path in [
        RANK_SWEEP / "hotpotqa_larger_model_3b_raw_n500_summary.json",
        RANK_SWEEP / "hotpotqa_larger_model_3b_support_first_n500_summary.json",
        RANK_SWEEP / "hotpotqa_larger_model_3b_gold_n500_summary.json",
    ]:
        for row in rows_from_summary(path, model_label="Llama-3.2-3B"):
            rows.append(
                round_row(
                    {
                        "model_label": "Llama-3.2-3B",
                        "interface_name": row["interface_name"],
                        "interface_label": INTERFACE_LABELS[row["interface_name"]],
                        "train_size": row["train_size"],
                        "updates": row["updates"],
                        "final_eval_loss": row["final_eval_loss"],
                        "exact_match": row["exact_match"],
                        "f1": row["f1"],
                        "answer_contained": row["answer_contained"],
                        "avg_prompt_tokens": row["avg_prompt_tokens"],
                        "truncated_count": row["truncated_count"],
                        "source_summary": row["source_summary"],
                    }
                )
            )

    order = {"Llama-3.2-1B": 0, "Llama-3.2-3B": 1}
    rows.sort(key=lambda r: (order.get(r["model_label"], 99), INTERFACE_ORDER.index(r["interface_name"])))
    return rows


def collect_larger_model_scaling_3b() -> List[Dict[str, Any]]:
    paths = [
        RANK_SWEEP / "hotpotqa_larger_model_3b_raw_n500_summary.json",
        RANK_SWEEP / "hotpotqa_larger_model_3b_support_first_n500_summary.json",
        RANK_SWEEP / "hotpotqa_larger_model_3b_gold_n500_summary.json",
        RANK_SWEEP / "hotpotqa_larger_model_3b_raw_n1000_summary.json",
        RANK_SWEEP / "hotpotqa_larger_model_3b_support_first_n1000_summary.json",
        RANK_SWEEP / "hotpotqa_larger_model_3b_gold_n1000_summary.json",
        RANK_SWEEP / "hotpotqa_lora_3b_raw_context_n2000_summary.json",
        RANK_SWEEP / "hotpotqa_lora_3b_raw_support_first_n2000_summary.json",
        RANK_SWEEP / "hotpotqa_lora_3b_gold_n2000_summary.json",
    ]
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for row in rows_from_summary(path, model_label="Llama-3.2-3B"):
            rows.append(
                round_row(
                    {
                        "model_label": "Llama-3.2-3B",
                        "interface_name": row["interface_name"],
                        "interface_label": INTERFACE_LABELS[row["interface_name"]],
                        "train_size": row["train_size"],
                        "updates": row["updates"],
                        "final_eval_loss": row["final_eval_loss"],
                        "exact_match": row["exact_match"],
                        "f1": row["f1"],
                        "answer_contained": row["answer_contained"],
                        "avg_prompt_tokens": row["avg_prompt_tokens"],
                        "truncated_count": row["truncated_count"],
                        "source_summary": row["source_summary"],
                    }
                )
            )
    interface_order = ["raw_context", "raw_support_first", "gold_supporting_sentences"]
    rows.sort(key=lambda r: (r["train_size"], interface_order.index(r["interface_name"])))
    return rows


def collect_position_stress_n500() -> List[Dict[str, Any]]:
    paths = [
        RANK_SWEEP / "hotpotqa_lora_rank_sweep_summary.json",
        RANK_SWEEP / "hotpotqa_position_shuffled_lora_r8_summary.json",
        RANK_SWEEP / "hotpotqa_position_middle_lora_r8_summary.json",
        RANK_SWEEP / "hotpotqa_position_stress_lora_r8_summary.json",
        RANK_SWEEP / "hotpotqa_evidence_quality_lora_r8_summary.json",
    ]
    wanted = {
        "raw_context",
        "raw_support_shuffled",
        "raw_support_middle",
        "raw_support_last",
        "raw_support_first",
    }
    dedup: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for row in rows_from_summary(path, model_label="Llama-3.2-1B"):
            if row.get("rank") != 8 or row.get("train_size") != 500:
                continue
            if row["interface_name"] not in wanted:
                continue
            dedup[row["interface_name"]] = row

    out = []
    for interface_name in [
        "raw_context",
        "raw_support_shuffled",
        "raw_support_middle",
        "raw_support_last",
        "raw_support_first",
    ]:
        row = dedup.get(interface_name)
        if not row:
            continue
        out.append(
            round_row(
                {
                    "model_label": "Llama-3.2-1B",
                    "interface_name": interface_name,
                    "interface_label": INTERFACE_LABELS[interface_name],
                    "train_size": row["train_size"],
                    "updates": row["updates"],
                    "final_eval_loss": row["final_eval_loss"],
                    "exact_match": row["exact_match"],
                    "f1": row["f1"],
                    "answer_contained": row["answer_contained"],
                    "avg_prompt_tokens": row["avg_prompt_tokens"],
                    "truncated_count": row["truncated_count"],
                    "source_summary": row["source_summary"],
                }
            )
        )
    return out


def collect_evidence_quality_n500() -> List[Dict[str, Any]]:
    rows = []
    for path in [
        RANK_SWEEP / "hotpotqa_lora_rank_sweep_summary.json",
        RANK_SWEEP / "hotpotqa_evidence_quality_lora_r8_summary.json",
    ]:
        rows.extend(rows_from_summary(path, model_label="Llama-3.2-1B"))

    wanted = {
        "raw_context",
        "raw_support_first",
        "gold_supporting_sentences",
        "gold_plus_1_distractor",
        "gold_plus_3_distractors",
    }
    out = []
    for row in rows:
        if row.get("rank") != 8 or row.get("train_size") != 500:
            continue
        if row["interface_name"] not in wanted:
            continue
        out.append(
            round_row(
                {
                    "model_label": "Llama-3.2-1B",
                    "interface_name": row["interface_name"],
                    "interface_label": INTERFACE_LABELS[row["interface_name"]],
                    "train_size": row["train_size"],
                    "final_eval_loss": row["final_eval_loss"],
                    "exact_match": row["exact_match"],
                    "f1": row["f1"],
                    "answer_contained": row["answer_contained"],
                    "avg_prompt_tokens": row["avg_prompt_tokens"],
                    "source_summary": row["source_summary"],
                }
            )
        )
    out.sort(key=lambda r: INTERFACE_ORDER.index(r["interface_name"]))
    return out


def collect_front_window_n500() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    paths = [
        RANK_SWEEP / "hotpotqa_lora_rank_sweep_summary.json",
        RANK_SWEEP / "hotpotqa_evidence_quality_lora_r8_summary.json",
        RANK_SWEEP / "hotpotqa_front_window_lora_r8_n500_summary.json",
    ]
    for path in paths:
        rows.extend(rows_from_summary(path, model_label="Llama-3.2-1B"))

    wanted_order = [
        "gold_supporting_sentences",
        "raw_support_first_2docs",
        "raw_support_first_3docs",
        "raw_support_first_5docs",
        "raw_support_first",
        "raw_context",
    ]
    dedup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("rank") != 8 or row.get("train_size") != 500:
            continue
        if row["interface_name"] not in wanted_order:
            continue
        dedup[row["interface_name"]] = row

    out = []
    for interface_name in wanted_order:
        row = dedup.get(interface_name)
        if not row:
            continue
        out.append(
            round_row(
                {
                    "model_label": "Llama-3.2-1B",
                    "interface_name": interface_name,
                    "interface_label": INTERFACE_LABELS[interface_name],
                    "train_size": row["train_size"],
                    "updates": row["updates"],
                    "final_eval_loss": row["final_eval_loss"],
                    "exact_match": row["exact_match"],
                    "f1": row["f1"],
                    "answer_contained": row["answer_contained"],
                    "avg_prompt_tokens": row["avg_prompt_tokens"],
                    "truncated_count": row["truncated_count"],
                    "source_summary": row["source_summary"],
                }
            )
        )
    return out


def collect_realistic_ordering_n500() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    paths = [
        RANK_SWEEP / "hotpotqa_lora_rank_sweep_summary.json",
        RANK_SWEEP / "hotpotqa_evidence_quality_lora_r8_summary.json",
        RANK_SWEEP / "hotpotqa_bm25_order_lora_r8_n500_summary.json",
        RANK_SWEEP / "hotpotqa_bm25_window_lora_r8_n500_summary.json",
        RANK_SWEEP / "hotpotqa_embed_retrieval_lora_r8_n500_summary.json",
        RANK_SWEEP / "hotpotqa_crossenc_retrieval_lora_r8_n500_summary.json",
        RANK_SWEEP / "hotpotqa_suptitle_retrieval_lora_r8_n500_summary.json",
        RANK_SWEEP / "hotpotqa_ftcrossenc_retrieval_lora_r8_n500_summary.json",
    ]
    for path in paths:
        rows.extend(rows_from_summary(path, model_label="Llama-3.2-1B"))

    wanted_order = [
        "raw_context",
        "raw_bm25_order",
        "raw_bm25_top3docs",
        "raw_bm25_top5docs",
        "raw_embed_order",
        "raw_embed_top3docs",
        "raw_embed_top5docs",
        "raw_crossenc_order",
        "raw_crossenc_top3docs",
        "raw_crossenc_top5docs",
        "raw_suptitle_order",
        "raw_suptitle_top3docs",
        "raw_suptitle_top5docs",
        "raw_ftcrossenc_order",
        "raw_ftcrossenc_top3docs",
        "raw_ftcrossenc_top5docs",
        "raw_support_first",
        "raw_support_first_2docs",
        "gold_supporting_sentences",
    ]
    dedup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("rank") != 8 or row.get("train_size") != 500:
            continue
        if row["interface_name"] not in wanted_order:
            continue
        dedup[row["interface_name"]] = row

    out = []
    for interface_name in wanted_order:
        row = dedup.get(interface_name)
        if not row:
            continue
        out.append(
            round_row(
                {
                    "model_label": "Llama-3.2-1B",
                    "interface_name": interface_name,
                    "interface_label": INTERFACE_LABELS[interface_name],
                    "train_size": row["train_size"],
                    "updates": row["updates"],
                    "final_eval_loss": row["final_eval_loss"],
                    "exact_match": row["exact_match"],
                    "f1": row["f1"],
                    "answer_contained": row["answer_contained"],
                    "avg_prompt_tokens": row["avg_prompt_tokens"],
                    "truncated_count": row["truncated_count"],
                    "source_summary": row["source_summary"],
                }
            )
        )
    return out


def collect_ftcrossenc_scaling_1b() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    paths = [
        RANK_SWEEP / "hotpotqa_ftcrossenc_retrieval_lora_r8_n500_summary.json",
        RANK_SWEEP / "hotpotqa_ftcrossenc_order_lora_r8_n1000_summary.json",
        RANK_SWEEP / "hotpotqa_ftcrossenc_top5docs_lora_r8_n1000_summary.json",
        RANK_SWEEP / "hotpotqa_ftcrossenc_order_lora_r8_n2000_summary.json",
        RANK_SWEEP / "hotpotqa_ftcrossenc_top5docs_lora_r8_n2000_summary.json",
    ]
    for path in paths:
        rows.extend(rows_from_summary(path, model_label="Llama-3.2-1B"))

    wanted_order = [
        "raw_ftcrossenc_order",
        "raw_ftcrossenc_top3docs",
        "raw_ftcrossenc_top5docs",
    ]
    out = []
    for row in rows:
        if row.get("rank") != 8:
            continue
        if row["interface_name"] not in wanted_order:
            continue
        out.append(
            round_row(
                {
                    "model_label": "Llama-3.2-1B",
                    "interface_name": row["interface_name"],
                    "interface_label": INTERFACE_LABELS[row["interface_name"]],
                    "train_size": row["train_size"],
                    "updates": row["updates"],
                    "final_eval_loss": row["final_eval_loss"],
                    "exact_match": row["exact_match"],
                    "f1": row["f1"],
                    "answer_contained": row["answer_contained"],
                    "avg_prompt_tokens": row["avg_prompt_tokens"],
                    "truncated_count": row["truncated_count"],
                    "source_summary": row["source_summary"],
                }
            )
        )
    out.sort(key=lambda r: (r["train_size"], wanted_order.index(r["interface_name"])))
    return out


def collect_ftcrossenc_model_check() -> List[Dict[str, Any]]:
    specs = [
        ("Llama-3.2-1B", RANK_SWEEP / "hotpotqa_ftcrossenc_retrieval_lora_r8_n500_summary.json"),
        ("Llama-3.2-1B", RANK_SWEEP / "hotpotqa_ftcrossenc_order_lora_r8_n1000_summary.json"),
        ("Llama-3.2-1B", RANK_SWEEP / "hotpotqa_ftcrossenc_order_lora_r8_n2000_summary.json"),
        ("Llama-3.2-1B", RANK_SWEEP / "hotpotqa_ftcrossenc_top5docs_lora_r8_n1000_summary.json"),
        ("Llama-3.2-1B", RANK_SWEEP / "hotpotqa_ftcrossenc_top5docs_lora_r8_n2000_summary.json"),
        ("Llama-3.2-3B", RANK_SWEEP / "hotpotqa_larger_model_3b_ftcrossenc_top5docs_n500_summary.json"),
        ("Llama-3.2-3B", RANK_SWEEP / "hotpotqa_larger_model_3b_ftcrossenc_top5docs_n1000_summary.json"),
        ("Llama-3.2-3B", RANK_SWEEP / "hotpotqa_larger_model_3b_ftcrossenc_top5docs_n2000_summary.json"),
        ("Llama-3.2-3B", RANK_SWEEP / "hotpotqa_larger_model_3b_ftcrossenc_order_n500_summary.json"),
        ("Llama-3.2-3B", RANK_SWEEP / "hotpotqa_larger_model_3b_ftcrossenc_order_n1000_summary.json"),
        ("Llama-3.2-3B", RANK_SWEEP / "hotpotqa_larger_model_3b_ftcrossenc_order_n2000_summary.json"),
    ]
    wanted = {"raw_ftcrossenc_top5docs", "raw_ftcrossenc_order"}
    rows: List[Dict[str, Any]] = []
    for model_label, path in specs:
        if not path.exists():
            continue
        for row in rows_from_summary(path, model_label=model_label):
            if row.get("rank") != 8:
                continue
            if row["interface_name"] not in wanted:
                continue
            rows.append(
                round_row(
                    {
                        "model_label": model_label,
                        "interface_name": row["interface_name"],
                        "interface_label": INTERFACE_LABELS[row["interface_name"]],
                        "train_size": row["train_size"],
                        "updates": row["updates"],
                        "final_eval_loss": row["final_eval_loss"],
                        "exact_match": row["exact_match"],
                        "f1": row["f1"],
                        "answer_contained": row["answer_contained"],
                        "avg_prompt_tokens": row["avg_prompt_tokens"],
                        "truncated_count": row["truncated_count"],
                        "source_summary": row["source_summary"],
                    }
                )
            )
    model_order = {"Llama-3.2-1B": 0, "Llama-3.2-3B": 1}
    interface_order = {"raw_ftcrossenc_top5docs": 0, "raw_ftcrossenc_order": 1}
    rows.sort(
        key=lambda r: (
            model_order.get(r["model_label"], 99),
            interface_order.get(r["interface_name"], 99),
            r["train_size"],
        )
    )
    return rows


def collect_subspace_runs() -> List[Dict[str, Any]]:
    specs = [
        SUBSPACE / "hotpotqa_hash_subspace_sweep_summary.json",
        SUBSPACE / "hotpotqa_hash_subspace_seed202_d65536_summary.json",
        SUBSPACE / "hotpotqa_hash_subspace_seed202_d262144_summary.json",
        SUBSPACE / "hotpotqa_hash_subspace_seed303_d65536_summary.json",
        SUBSPACE / "hotpotqa_hash_subspace_seed303_d262144_summary.json",
        SUBSPACE / "hotpotqa_hash_subspace_raw_seed101_d524288_summary.json",
        SUBSPACE / "hotpotqa_evidence_quality_hash_seed101_summary.json",
        SUBSPACE / "hotpotqa_evidence_quality_hash_seed202_summary.json",
    ]
    out = []
    seen: set[tuple[str, int, int]] = set()
    for path in specs:
        data = load_json(path)
        seed = int(data["seed"])
        for row in data["rows"]:
            key = (row["interface_name"], int(row["active_dim"]), seed)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                round_row(
                    {
                        "model_label": "Llama-3.2-1B",
                        "interface_name": row["interface_name"],
                        "interface_label": INTERFACE_LABELS.get(row["interface_name"], row["interface_name"]),
                        "method": row.get("method", data.get("method")),
                        "seed": seed,
                        "active_dim": row["active_dim"],
                        "used_dim": row.get("used_dim"),
                        "train_size": row["train_size"],
                        "updates": row["updates"],
                        "final_eval_loss": row["final_eval_loss"],
                        "exact_match": row["exact_match"],
                        "f1": row["f1"],
                        "answer_contained": row["answer_contained"],
                        "source_summary": str(path.relative_to(OUTPUTS)),
                    }
                )
            )
    out.sort(key=lambda r: (INTERFACE_ORDER.index(r["interface_name"]) if r["interface_name"] in INTERFACE_ORDER else 99, r["active_dim"], r["seed"]))
    return out


def aggregate_subspace_runs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["interface_name"], int(row["active_dim"]))].append(row)

    out = []
    for (interface_name, active_dim), group in groups.items():
        ems = [r["exact_match"] for r in group]
        f1s = [r["f1"] for r in group]
        losses = [r["final_eval_loss"] for r in group]
        out.append(
            round_row(
                {
                    "model_label": "Llama-3.2-1B",
                    "interface_name": interface_name,
                    "interface_label": INTERFACE_LABELS.get(interface_name, interface_name),
                    "active_dim": active_dim,
                    "n": len(group),
                    "seeds": ",".join(str(r["seed"]) for r in sorted(group, key=lambda r: r["seed"])),
                    "eval_loss_mean": mean(losses),
                    "em_mean": mean(ems),
                    "em_sd": stdev(ems) if len(ems) > 1 else 0.0,
                    "f1_mean": mean(f1s),
                    "f1_sd": stdev(f1s) if len(f1s) > 1 else 0.0,
                }
            )
        )
    out.sort(key=lambda r: (INTERFACE_ORDER.index(r["interface_name"]) if r["interface_name"] in INTERFACE_ORDER else 99, r["active_dim"]))
    return out


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 200,
        }
    )


def savefig(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_scaling_f1(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for interface_name in ["raw_context", "raw_support_first", "gold_supporting_sentences"]:
        group = [r for r in rows if r["interface_name"] == interface_name]
        group.sort(key=lambda r: r["train_size"])
        ax.plot(
            [r["train_size"] for r in group],
            [r["f1"] for r in group],
            marker="o",
            linewidth=2,
            color=COLORS[interface_name],
            label=INTERFACE_LABELS[interface_name],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([500, 1000, 2000, 4000], labels=["500", "1000", "2000", "4000"])
    ax.set_ylim(0.34, 0.76)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1")
    ax.set_title("Full-LoRA data scaling on Llama-3.2-1B")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, out_dir / "data_scaling_f1_1b")


def plot_scaling_em(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for interface_name in ["raw_context", "raw_support_first", "gold_supporting_sentences"]:
        group = [r for r in rows if r["interface_name"] == interface_name]
        group.sort(key=lambda r: r["train_size"])
        ax.plot(
            [r["train_size"] for r in group],
            [r["exact_match"] for r in group],
            marker="o",
            linewidth=2,
            color=COLORS[interface_name],
            label=INTERFACE_LABELS[interface_name],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([500, 1000, 2000, 4000], labels=["500", "1000", "2000", "4000"])
    ax.set_ylim(0.24, 0.62)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Exact match")
    ax.set_title("Full-LoRA data scaling on Llama-3.2-1B")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, out_dir / "data_scaling_em_1b")


def plot_gap_decomposition(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    by_size: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_size[row["train_size"]][row["interface_name"]] = row

    xs, support_minus_raw, gold_minus_support = [], [], []
    for train_size in [500, 1000, 2000, 4000]:
        group = by_size[train_size]
        if not all(k in group for k in ["raw_context", "raw_support_first", "gold_supporting_sentences"]):
            continue
        xs.append(train_size)
        support_minus_raw.append(group["raw_support_first"]["f1"] - group["raw_context"]["f1"])
        gold_minus_support.append(group["gold_supporting_sentences"]["f1"] - group["raw_support_first"]["f1"])

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.34
    positions = range(len(xs))
    ax.bar([p - width / 2 for p in positions], support_minus_raw, width=width, color="#10b981", label="Support first - raw")
    ax.bar([p + width / 2 for p in positions], gold_minus_support, width=width, color="#6b7280", label="Gold - support first")
    ax.set_xticks(list(positions), labels=[str(x) for x in xs])
    ax.set_ylim(0, 0.18)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1 gap")
    ax.set_title("Where the raw-context gap comes from")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    savefig(fig, out_dir / "ordering_gap_decomposition_f1_1b")


def plot_larger_model(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    interfaces = ["raw_context", "raw_support_first", "gold_supporting_sentences"]
    models = ["Llama-3.2-1B", "Llama-3.2-3B"]
    width = 0.34
    positions = range(len(interfaces))
    lookup = {(r["model_label"], r["interface_name"]): r for r in rows}
    for offset, model in [(-width / 2, models[0]), (width / 2, models[1])]:
        ax.bar(
            [p + offset for p in positions],
            [lookup[(model, i)]["f1"] for i in interfaces],
            width=width,
            label=model,
            color="#94a3b8" if model.endswith("1B") else "#2563eb",
        )
    ax.set_xticks(list(positions), labels=[INTERFACE_LABELS[i] for i in interfaces])
    ax.set_ylim(0.32, 0.78)
    ax.set_ylabel("F1")
    ax.set_title("Larger-model sanity check at 500 examples")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    savefig(fig, out_dir / "larger_model_n500_f1")


def plot_larger_model_scaling_3b(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for interface_name in ["raw_context", "raw_support_first", "gold_supporting_sentences"]:
        group = [r for r in rows if r["interface_name"] == interface_name]
        group.sort(key=lambda r: r["train_size"])
        ax.plot(
            [r["train_size"] for r in group],
            [r["f1"] for r in group],
            marker="o",
            linewidth=2,
            color=COLORS[interface_name],
            label=INTERFACE_LABELS[interface_name],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([500, 1000, 2000], labels=["500", "1000", "2000"])
    ax.set_ylim(0.52, 0.80)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("F1")
    ax.set_title("Full-LoRA scaling on Llama-3.2-3B")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, out_dir / "larger_model_scaling_3b_f1")


def plot_position_stress(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    rows = sorted(rows, key=lambda r: INTERFACE_ORDER.index(r["interface_name"]))
    positions = range(len(rows))
    ax.bar(
        list(positions),
        [r["f1"] for r in rows],
        color=[COLORS[r["interface_name"]] for r in rows],
        width=0.64,
    )
    ax.set_xticks(list(positions), labels=[r["interface_label"] for r in rows], rotation=20, ha="right")
    ax.set_ylim(0.34, 0.57)
    ax.set_ylabel("F1")
    ax.set_title("Support-position stress test on Llama-3.2-1B")
    ax.grid(True, axis="y", alpha=0.25)
    savefig(fig, out_dir / "position_stress_f1_1b")


def plot_front_window(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    positions = range(len(rows))
    ax.bar(
        list(positions),
        [r["f1"] for r in rows],
        color=[COLORS[r["interface_name"]] for r in rows],
        width=0.64,
    )
    ax.set_xticks(list(positions), labels=[r["interface_label"] for r in rows], rotation=22, ha="right")
    ax.set_ylim(0.34, 0.63)
    ax.set_ylabel("F1")
    ax.set_title("Support-first front-window ablation on Llama-3.2-1B")
    ax.grid(True, axis="y", alpha=0.25)
    savefig(fig, out_dir / "front_window_f1_1b")


def plot_realistic_ordering(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    positions = range(len(rows))
    ax.bar(
        list(positions),
        [r["f1"] for r in rows],
        color=[COLORS[r["interface_name"]] for r in rows],
        width=0.62,
    )
    ax.set_xticks(list(positions), labels=[r["interface_label"] for r in rows], rotation=18, ha="right")
    ax.set_ylim(0.34, 0.63)
    ax.set_ylabel("F1")
    ax.set_title("Realistic ordering check on Llama-3.2-1B")
    ax.grid(True, axis="y", alpha=0.25)
    savefig(fig, out_dir / "realistic_ordering_f1_1b")


def plot_subspace(agg_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for interface_name in ["raw_context", "raw_support_first", "gold_plus_3_distractors", "gold_supporting_sentences"]:
        group = [r for r in agg_rows if r["interface_name"] == interface_name]
        group.sort(key=lambda r: r["active_dim"])
        if not group:
            continue
        ax.errorbar(
            [r["active_dim"] for r in group],
            [r["f1_mean"] for r in group],
            yerr=[r["f1_sd"] for r in group],
            marker="o",
            linewidth=2,
            capsize=3,
            color=COLORS.get(interface_name),
            label=INTERFACE_LABELS.get(interface_name, interface_name),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([65536, 131072, 262144, 524288], labels=["65k", "131k", "262k", "524k"])
    ax.set_ylim(0.18, 0.56)
    ax.set_xlabel("Active hash subspace dimension")
    ax.set_ylabel("F1")
    ax.set_title("Hash-subspace capacity on Llama-3.2-1B")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    savefig(fig, out_dir / "hash_subspace_f1")


def main() -> None:
    args = parse_args()
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    setup_plot()

    scaling_rows = collect_full_lora_scaling_1b()
    larger_rows = collect_larger_model_n500(scaling_rows)
    larger_scaling_3b = collect_larger_model_scaling_3b()
    position_rows = collect_position_stress_n500()
    evidence_rows = collect_evidence_quality_n500()
    front_window_rows = collect_front_window_n500()
    realistic_ordering_rows = collect_realistic_ordering_n500()
    ftcrossenc_scaling_rows = collect_ftcrossenc_scaling_1b()
    ftcrossenc_model_rows = collect_ftcrossenc_model_check()
    subspace_runs = collect_subspace_runs()
    subspace_agg = aggregate_subspace_runs(subspace_runs)

    metric_fields = [
        "model_label",
        "interface_name",
        "interface_label",
        "train_size",
        "updates",
        "final_eval_loss",
        "exact_match",
        "f1",
        "answer_contained",
        "avg_prompt_tokens",
        "truncated_count",
        "source_summary",
    ]
    write_csv(table_dir / "full_lora_scaling_1b.csv", scaling_rows, metric_fields)
    write_json(table_dir / "full_lora_scaling_1b.json", scaling_rows)

    write_csv(table_dir / "larger_model_n500.csv", larger_rows, metric_fields)
    write_json(table_dir / "larger_model_n500.json", larger_rows)

    write_csv(table_dir / "larger_model_scaling_3b.csv", larger_scaling_3b, metric_fields)
    write_json(table_dir / "larger_model_scaling_3b.json", larger_scaling_3b)

    write_csv(table_dir / "position_stress_n500.csv", position_rows, metric_fields)
    write_json(table_dir / "position_stress_n500.json", position_rows)

    write_csv(table_dir / "front_window_n500.csv", front_window_rows, metric_fields)
    write_json(table_dir / "front_window_n500.json", front_window_rows)

    write_csv(table_dir / "realistic_ordering_n500.csv", realistic_ordering_rows, metric_fields)
    write_json(table_dir / "realistic_ordering_n500.json", realistic_ordering_rows)

    write_csv(table_dir / "ftcrossenc_scaling_1b.csv", ftcrossenc_scaling_rows, metric_fields)
    write_json(table_dir / "ftcrossenc_scaling_1b.json", ftcrossenc_scaling_rows)

    write_csv(table_dir / "ftcrossenc_model_check.csv", ftcrossenc_model_rows, metric_fields)
    write_json(table_dir / "ftcrossenc_model_check.json", ftcrossenc_model_rows)

    evidence_fields = [
        "model_label",
        "interface_name",
        "interface_label",
        "train_size",
        "final_eval_loss",
        "exact_match",
        "f1",
        "answer_contained",
        "avg_prompt_tokens",
        "source_summary",
    ]
    write_csv(table_dir / "evidence_quality_n500.csv", evidence_rows, evidence_fields)
    write_json(table_dir / "evidence_quality_n500.json", evidence_rows)

    subspace_fields = [
        "model_label",
        "interface_name",
        "interface_label",
        "method",
        "seed",
        "active_dim",
        "used_dim",
        "train_size",
        "updates",
        "final_eval_loss",
        "exact_match",
        "f1",
        "answer_contained",
        "source_summary",
    ]
    write_csv(table_dir / "hash_subspace_runs.csv", subspace_runs, subspace_fields)
    write_json(table_dir / "hash_subspace_runs.json", subspace_runs)

    subspace_agg_fields = [
        "model_label",
        "interface_name",
        "interface_label",
        "active_dim",
        "n",
        "seeds",
        "eval_loss_mean",
        "em_mean",
        "em_sd",
        "f1_mean",
        "f1_sd",
    ]
    write_csv(table_dir / "hash_subspace_aggregate.csv", subspace_agg, subspace_agg_fields)
    write_json(table_dir / "hash_subspace_aggregate.json", subspace_agg)

    plot_scaling_f1(scaling_rows, figure_dir)
    plot_scaling_em(scaling_rows, figure_dir)
    plot_gap_decomposition(scaling_rows, figure_dir)
    plot_larger_model(larger_rows, figure_dir)
    plot_larger_model_scaling_3b(larger_scaling_3b, figure_dir)
    plot_position_stress(position_rows, figure_dir)
    plot_front_window(front_window_rows, figure_dir)
    plot_realistic_ordering(realistic_ordering_rows, figure_dir)
    plot_subspace(subspace_agg, figure_dir)

    manifest = {
        "tables": sorted(str(p.relative_to(args.output_dir)) for p in table_dir.glob("*")),
        "figures": sorted(str(p.relative_to(args.output_dir)) for p in figure_dir.glob("*")),
    }
    write_json(args.output_dir / "manifest.json", manifest)

    print(f"wrote consolidated outputs to {args.output_dir}")
    print("tables:")
    for item in manifest["tables"]:
        print(f"- {item}")
    print("figures:")
    for item in manifest["figures"]:
        print(f"- {item}")


if __name__ == "__main__":
    main()
