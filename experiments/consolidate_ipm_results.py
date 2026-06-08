#!/usr/bin/env python
"""Build IPM-ready result tables and figures from rank-sweep summaries."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
OUTPUTS = EXPERIMENTS / "outputs"
RANK_SWEEP = OUTPUTS / "rank_sweep"
DEFAULT_OUT = OUTPUTS / "consolidated"
DEFAULT_PAPER_FIG_DIR = ROOT / "paper_ipm" / "figs"


DATASET_LABELS = {
    "hotpotqa": "HotpotQA",
    "2wiki": "2WikiMultiHopQA",
    "musique": "MuSiQue",
}

DATASET_ORDER = ["hotpotqa", "2wiki", "musique"]

INTERFACE_LABELS = {
    "no_context": "No context",
    "raw_context": "Raw context",
    "raw_support_first": "Support first",
    "raw_support_middle": "Support middle",
    "raw_support_last": "Support last",
    "raw_support_shuffled": "Support shuffled",
    "raw_bm25_order": "BM25 order",
    "raw_bm25_top5docs": "BM25 top-5",
    "raw_crossenc_order": "Cross-enc order",
    "raw_crossenc_top3docs": "Cross-enc top-3",
    "raw_crossenc_top5docs": "Cross-enc top-5",
    "raw_ftcrossenc_order": "FT cross-enc order",
    "raw_ftcrossenc_top3docs": "FT cross-enc top-3",
    "raw_ftcrossenc_top5docs": "FT cross-enc top-5",
    "gold_supporting_sentences": "Oracle support",
    "gold_supporting_paragraphs": "Oracle support",
    "gold_evidence_triples": "Structured triples",
}

MAIN_INTERFACE_ORDER = [
    "raw_context",
    "raw_support_first",
    "realistic_top5",
    "oracle_support",
    "structured",
]

MAIN_INTERFACE_LABELS = {
    "raw_context": "Raw context",
    "raw_support_first": "Support first",
    "realistic_top5": "Realistic top-5",
    "oracle_support": "Oracle support",
    "structured": "Structured triples",
}

N2000_INTERFACE_ORDER = [
    "no_context",
    "raw_context",
    "realistic_top5",
    "raw_support_first",
    "oracle_support",
    "structured",
]

N2000_INTERFACE_LABELS = {
    "no_context": "Question only",
    "raw_context": "Raw context",
    "realistic_top5": "Top-5 window",
    "raw_support_first": "Support first",
    "oracle_support": "Oracle support",
    "structured": "Structured triples",
}

COLORS = {
    "no_context": "#94a3b8",
    "raw_context": "#0f766e",
    "raw_support_first": "#f97316",
    "realistic_top5": "#2563eb",
    "oracle_support": "#16a34a",
    "structured": "#7c3aed",
    "gain": "#f97316",
    "median": "#334155",
    "grid": "#e2e8f0",
    "range": "#d9e2ec",
}

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@dataclass(frozen=True)
class MainSpec:
    dataset: str
    interface_group: str
    path: str


QWEN14B_N4000_SPECS = [
    MainSpec("hotpotqa", "raw_context", "hotpotqa_qwen14b_r8_n4000_raw_context_summary.json"),
    MainSpec("hotpotqa", "raw_support_first", "hotpotqa_qwen14b_r8_n4000_raw_support_first_summary.json"),
    MainSpec("hotpotqa", "realistic_top5", "hotpotqa_qwen14b_r8_n4000_raw_ftcrossenc_top5docs_summary.json"),
    MainSpec("hotpotqa", "oracle_support", "hotpotqa_qwen14b_r8_n4000_gold_supporting_sentences_summary.json"),
    MainSpec("2wiki", "raw_context", "2wiki_qwen14b_r8_n4000_raw_context_summary.json"),
    MainSpec("2wiki", "raw_support_first", "2wiki_qwen14b_r8_n4000_raw_support_first_summary.json"),
    MainSpec("2wiki", "realistic_top5", "2wiki_qwen14b_r8_n4000_raw_crossenc_top5docs_summary.json"),
    MainSpec("2wiki", "oracle_support", "2wiki_qwen14b_r8_n4000_gold_supporting_sentences_summary.json"),
    MainSpec("2wiki", "structured", "2wiki_qwen14b_r8_n4000_gold_evidence_triples_summary.json"),
    MainSpec("musique", "raw_context", "musique_qwen14b_r8_n4000_raw_context_summary.json"),
    MainSpec("musique", "raw_support_first", "musique_qwen14b_r8_n4000_raw_support_first_summary.json"),
    MainSpec("musique", "realistic_top5", "musique_qwen14b_r8_n4000_raw_crossenc_top5docs_summary.json"),
    MainSpec("musique", "oracle_support", "musique_qwen14b_r8_n4000_gold_supporting_paragraphs_summary.json"),
]

QWEN14B_N2000_CONTROL_PATTERNS = [
    "hotpotqa_qwen14b_r8_n2000_*_summary.json",
    "2wiki_qwen14b_r8_n2000_*_summary.json",
    "musique_qwen14b_r8_n2000_*_summary.json",
]

CROSS_MODEL_N1000_FILES = [
    "hotpotqa_qwen7b_r8_n1000_no_context_summary.json",
    "hotpotqa_qwen7b_r8_n1000_raw_context_summary.json",
    "hotpotqa_qwen7b_r8_n1000_raw_support_first_summary.json",
    "hotpotqa_qwen7b_r8_n1000_raw_ftcrossenc_top5docs_summary.json",
    "hotpotqa_qwen7b_r8_n1000_gold_supporting_sentences_summary.json",
    "2wiki_qwen7b_r8_n1000_no_context_summary.json",
    "2wiki_qwen7b_r8_n1000_raw_context_summary.json",
    "2wiki_qwen7b_r8_n1000_raw_support_first_summary.json",
    "2wiki_qwen7b_r8_n1000_raw_crossenc_top5docs_summary.json",
    "2wiki_qwen7b_r8_n1000_gold_supporting_sentences_summary.json",
    "2wiki_qwen7b_r8_n1000_gold_evidence_triples_summary.json",
    "musique_qwen7b_r8_n1000_no_context_summary.json",
    "musique_qwen7b_r8_n1000_raw_context_summary.json",
    "musique_qwen7b_r8_n1000_raw_support_first_summary.json",
    "musique_qwen7b_r8_n1000_raw_crossenc_top5docs_summary.json",
    "musique_qwen7b_r8_n1000_gold_supporting_paragraphs_summary.json",
    "hotpotqa_mistral7b_r8_n1000_long_interfaces_summary.json",
    "hotpotqa_mistral7b_r8_n1000_short_interfaces_summary.json",
    "2wiki_mistral7b_r8_n1000_long_interfaces_summary.json",
    "2wiki_mistral7b_r8_n1000_short_interfaces_summary.json",
    "musique_mistral7b_r8_n1000_long_interfaces_summary.json",
    "musique_mistral7b_r8_n1000_short_interfaces_summary.json",
    "hotpotqa_llama31_8b_r8_n1000_long_interfaces_summary.json",
    "hotpotqa_llama31_8b_r8_n1000_short_interfaces_summary.json",
    "2wiki_llama31_8b_r8_n1000_long_interfaces_summary.json",
    "2wiki_llama31_8b_r8_n1000_short_interfaces_summary.json",
    "musique_llama31_8b_r8_n1000_long_interfaces_summary.json",
    "musique_llama31_8b_r8_n1000_short_interfaces_summary.json",
    "hotpotqa_gemma2_9b_r8_n1000_long_interfaces_summary.json",
    "hotpotqa_gemma2_9b_r8_n1000_short_interfaces_summary.json",
    "2wiki_gemma2_9b_r8_n1000_long_interfaces_summary.json",
    "2wiki_gemma2_9b_r8_n1000_short_interfaces_summary.json",
    "musique_gemma2_9b_r8_n1000_short_interfaces_summary.json",
    "musique_gemma2_9b_r8_n1000_raw_support_first_retry_summary.json",
]

CORE_N4000_FILES = [
    "hotpotqa_3b_r8_n4000_raw_context_summary.json",
    "hotpotqa_3b_r8_n4000_raw_support_first_summary.json",
    "hotpotqa_3b_r8_n4000_gold_supporting_sentences_summary.json",
    "2wiki_1b_r8_n4000_core_summary.json",
    "2wiki_3b_r8_n4000_core_summary.json",
    "musique_1b_r8_n4000_core_summary.json",
    "musique_3b_r8_n4000_core_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--paper-fig-dir", type=Path, default=DEFAULT_PAPER_FIG_DIR)
    parser.add_argument("--skip-paper-copy", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def infer_dataset(path: Path) -> str:
    name = path.name
    for dataset in DATASET_ORDER:
        if name.startswith(f"{dataset}_"):
            return dataset
    raise ValueError(f"Cannot infer dataset from {path}")


def model_label(model_id: str | None, path: Path | None = None) -> str:
    model_id = model_id or ""
    name = path.name if path else ""
    if "Qwen2.5-Coder-32B" in model_id or "qwen32b" in name:
        return "Qwen2.5-Coder-32B"
    if "Qwen2.5-Coder-14B" in model_id or "qwen14b" in name:
        return "Qwen2.5-Coder-14B"
    if "Qwen2.5-Coder-7B" in model_id or "qwen7b" in name:
        return "Qwen2.5-Coder-7B"
    if "Mistral-7B" in model_id or "mistral7b" in name:
        return "Mistral-7B"
    if "Llama-3.1-8B" in model_id or "llama31_8b" in name:
        return "Llama-3.1-8B"
    if "gemma-2-9b" in model_id or "gemma2_9b" in name:
        return "Gemma-2-9B"
    if "Llama-3.2-3B" in model_id or "_3b_" in name:
        return "Llama-3.2-3B"
    if "Llama-3.2-1B" in model_id or "_1b_" in name:
        return "Llama-3.2-1B"
    if model_id:
        return model_id.rsplit("/", 1)[-1]
    return "unknown"


def round_float(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def rows_from_summary(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    dataset = infer_dataset(path)
    rows = []
    for row in data.get("rows", []):
        normalized = dict(row)
        normalized["dataset"] = dataset
        normalized["dataset_label"] = DATASET_LABELS[dataset]
        normalized["model_id"] = data.get("model_id")
        normalized["model_label"] = model_label(data.get("model_id"), path)
        normalized["interface_label"] = INTERFACE_LABELS.get(
            normalized.get("interface_name", ""), normalized.get("interface_name", "")
        )
        normalized["source_summary"] = str(path.relative_to(ROOT))
        rows.append(normalized)
    return rows


def read_required(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return rows_from_summary(path)


def select_result(row: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "dataset": row["dataset"],
        "dataset_label": row["dataset_label"],
        "model_label": row["model_label"],
        "model_id": row.get("model_id", ""),
        "interface_name": row["interface_name"],
        "interface_label": row["interface_label"],
        "train_size": row.get("train_size", ""),
        "rank": row.get("rank", ""),
        "updates": row.get("updates", ""),
        "exact_match": round_float(row.get("exact_match")),
        "f1": round_float(row.get("f1")),
        "answer_contained": round_float(row.get("answer_contained")),
        "avg_prompt_tokens": round_float(row.get("avg_prompt_tokens"), 2),
        "truncated_count": row.get("truncated_count", ""),
        "source_summary": row.get("source_summary", ""),
    }
    if extra:
        out.update(extra)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, title: str, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"## {title}", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_all_rank_sweep_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RANK_SWEEP.glob("*summary.json")):
        try:
            for row in rows_from_summary(path):
                rows.append(select_result(row))
        except Exception as exc:
            rows.append(
                {
                    "dataset": "",
                    "dataset_label": "",
                    "model_label": "",
                    "model_id": "",
                    "interface_name": "",
                    "interface_label": "",
                    "train_size": "",
                    "rank": "",
                    "updates": "",
                    "exact_match": "",
                    "f1": "",
                    "answer_contained": "",
                    "avg_prompt_tokens": "",
                    "truncated_count": "",
                    "source_summary": str(path.relative_to(ROOT)),
                    "error": str(exc),
                }
            )
    return rows


def collect_qwen14b_n4000() -> list[dict[str, Any]]:
    rows = []
    for spec in QWEN14B_N4000_SPECS:
        path = RANK_SWEEP / spec.path
        row = read_required(path)[0]
        if row["dataset"] != spec.dataset:
            raise ValueError(f"Dataset mismatch for {path}")
        rows.append(
            select_result(
                row,
                {
                    "interface_group": spec.interface_group,
                    "interface_group_label": MAIN_INTERFACE_LABELS[spec.interface_group],
                    "f1_gain_over_raw": "",
                },
            )
        )

    raw_by_dataset = {
        row["dataset"]: row["f1"]
        for row in rows
        if row["interface_group"] == "raw_context"
    }
    for row in rows:
        raw = raw_by_dataset.get(row["dataset"])
        if isinstance(raw, float) and isinstance(row["f1"], float):
            row["f1_gain_over_raw"] = round(row["f1"] - raw, 6)
    rows.sort(key=lambda r: (DATASET_ORDER.index(r["dataset"]), MAIN_INTERFACE_ORDER.index(r["interface_group"])))
    return rows


def collect_qwen14b_n2000_controls() -> list[dict[str, Any]]:
    paths: list[Path] = []
    for pattern in QWEN14B_N2000_CONTROL_PATTERNS:
        paths.extend(RANK_SWEEP.glob(pattern))
    rows = []
    for path in sorted(set(paths)):
        for row in rows_from_summary(path):
            rows.append(select_result(row))
    rows.sort(key=lambda r: (DATASET_ORDER.index(r["dataset"]), r["interface_name"]))
    return rows


def collect_cross_model_n1000() -> list[dict[str, Any]]:
    rows = []
    for filename in CROSS_MODEL_N1000_FILES:
        path = RANK_SWEEP / filename
        if not path.exists():
            continue
        for row in rows_from_summary(path):
            rows.append(select_result(row))

    for filename in CORE_N4000_FILES:
        path = RANK_SWEEP / filename
        if not path.exists():
            continue
        for row in rows_from_summary(path):
            rows.append(select_result(row, {"note": "n4000 small-reader core"}))

    rows.sort(key=lambda r: (DATASET_ORDER.index(r["dataset"]), r["model_label"], r["train_size"], r["interface_name"]))
    return rows


def collect_qwen14b_n2000_no_context_gap(control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    realistic_by_dataset = {
        "hotpotqa": "raw_ftcrossenc_top5docs",
        "2wiki": "raw_crossenc_top5docs",
        "musique": "raw_crossenc_top5docs",
    }
    oracle_by_dataset = {
        "hotpotqa": "gold_supporting_sentences",
        "2wiki": "gold_supporting_sentences",
        "musique": "gold_supporting_paragraphs",
    }
    grouped: dict[tuple[str, str], dict[str, Any]] = {
        (row["dataset"], row["interface_name"]): row
        for row in control_rows
        if row["model_label"] == "Qwen2.5-Coder-14B" and int(row["train_size"]) == 2000
    }

    out = []
    for dataset in DATASET_ORDER:
        selected = {
            "no_context": "no_context",
            "raw_context": "raw_context",
            "realistic_top5": realistic_by_dataset[dataset],
            "raw_support_first": "raw_support_first",
            "oracle_support": oracle_by_dataset[dataset],
        }
        if dataset == "2wiki":
            selected["structured"] = "gold_evidence_triples"
        for group in N2000_INTERFACE_ORDER:
            interface_name = selected.get(group)
            if not interface_name:
                continue
            row = grouped.get((dataset, interface_name))
            if not row:
                continue
            normalized = dict(row)
            normalized["interface_group"] = group
            normalized["interface_group_label"] = N2000_INTERFACE_LABELS[group]
            out.append(normalized)
    out.sort(key=lambda r: (DATASET_ORDER.index(r["dataset"]), N2000_INTERFACE_ORDER.index(r["interface_group"])))
    return out


def pivot_main_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["dataset"], row["interface_group"]): row for row in rows}
    table = []
    for dataset in DATASET_ORDER:
        out: dict[str, Any] = {"dataset_label": DATASET_LABELS[dataset]}
        for group in MAIN_INTERFACE_ORDER:
            row = by_key.get((dataset, group))
            out[group] = row["f1"] if row else ""
            out[f"{group}_tokens"] = row["avg_prompt_tokens"] if row else ""
        table.append(out)
    return table


def support_gain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["dataset"], row["model_label"], int(row["train_size"]))
        grouped.setdefault(key, {})[row["interface_name"]] = row

    out = []
    for (dataset, model, train_size), interfaces in grouped.items():
        raw = interfaces.get("raw_context")
        support = interfaces.get("raw_support_first")
        if not raw or not support:
            continue
        out.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "model_label": model,
                "train_size": train_size,
                "raw_context_f1": raw["f1"],
                "support_first_f1": support["f1"],
                "support_gain_f1": round(support["f1"] - raw["f1"], 6),
                "raw_tokens": raw["avg_prompt_tokens"],
                "support_tokens": support["avg_prompt_tokens"],
            }
        )
    model_order = {
        "Llama-3.2-1B": 0,
        "Llama-3.2-3B": 1,
        "Qwen2.5-Coder-7B": 2,
        "Mistral-7B": 3,
        "Llama-3.1-8B": 4,
        "Gemma-2-9B": 5,
        "Qwen2.5-Coder-14B": 6,
        "Qwen2.5-Coder-32B": 7,
    }
    out.sort(key=lambda r: (DATASET_ORDER.index(r["dataset"]), model_order.get(r["model_label"], 99), r["train_size"]))
    return out


def plot_qwen14b_interface_hierarchy(rows: list[dict[str, Any]], outdir: Path, paper_fig_dir: Path | None) -> list[Path]:
    by_key = {(row["dataset"], row["interface_group"]): row for row in rows}
    groups = ["raw_context", "raw_support_first", "realistic_top5", "oracle_support"]
    x_positions = list(range(len(DATASET_ORDER)))
    markers = {
        "raw_context": "o",
        "raw_support_first": "D",
        "realistic_top5": "s",
        "oracle_support": "^",
    }
    line_styles = {
        "raw_context": "-",
        "raw_support_first": "-",
        "realistic_top5": "--",
        "oracle_support": "-",
    }

    fig, ax = plt.subplots(figsize=(5.7, 2.45))
    for group in groups:
        ys = [by_key[(dataset, group)]["f1"] for dataset in DATASET_ORDER]
        ax.plot(
            x_positions,
            ys,
            color=COLORS[group],
            linestyle=line_styles[group],
            linewidth=1.7,
            marker=markers[group],
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=MAIN_INTERFACE_LABELS[group],
            zorder=3,
        )
        ax.text(
            x_positions[-1] + 0.08,
            ys[-1],
            compact_interface_label(group),
            ha="left",
            va="center",
            fontsize=7.6,
            color=COLORS[group],
        )

    structured = by_key.get(("2wiki", "structured"))
    if structured:
        ax.scatter(
            [x_positions[1]],
            [structured["f1"]],
            marker="D",
            s=58,
            color=COLORS["structured"],
            edgecolor="white",
            linewidth=0.6,
            label="2Wiki structured triples",
            zorder=4,
        )
        ax.text(
            x_positions[1] + 0.08,
            structured["f1"],
            "Structured triples",
            ha="left",
            va="center",
            fontsize=7.6,
            color=COLORS["structured"],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.set_ylabel("Answer F1")
    ax.set_ylim(0.42, 1.02)
    ax.set_xlim(-0.12, len(DATASET_ORDER) - 0.45)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, linestyle="-", alpha=0.95)
    fig.tight_layout()

    paths = save_figure(fig, outdir, "ipm_fig1_qwen14b_interface_hierarchy")
    copy_to_paper(paths, paper_fig_dir)
    return paths


def plot_support_gain(rows: list[dict[str, Any]], outdir: Path, paper_fig_dir: Path | None) -> list[Path]:
    focus_models = [
        "Llama-3.2-1B",
        "Llama-3.2-3B",
        "Qwen2.5-Coder-7B",
        "Mistral-7B",
        "Llama-3.1-8B",
        "Gemma-2-9B",
        "Qwen2.5-Coder-14B",
        "Qwen2.5-Coder-32B",
    ]
    filtered = [row for row in rows if row["model_label"] in focus_models]

    fig, ax = plt.subplots(figsize=(4.85, 2.45))
    xticks = list(range(len(DATASET_ORDER)))
    for dataset_idx, dataset in enumerate(DATASET_ORDER):
        data = [row for row in filtered if row["dataset"] == dataset]
        data.sort(key=lambda r: focus_models.index(r["model_label"]))
        if not data:
            continue
        gains = [row["support_gain_f1"] for row in data]
        offsets = centered_offsets(len(data), span=0.24)
        xs = [dataset_idx + offset for offset in offsets]
        ax.scatter(
            xs,
            gains,
            s=38,
            color=COLORS["gain"],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        median_gain = sorted(gains)[len(gains) // 2] if len(gains) % 2 else sum(sorted(gains)[len(gains)//2-1:len(gains)//2+1]) / 2
        ax.plot(
            [dataset_idx - 0.18, dataset_idx + 0.18],
            [median_gain, median_gain],
            color=COLORS["median"],
            linewidth=1.25,
            solid_capstyle="round",
            zorder=4,
        )
        ax.text(
            dataset_idx,
            max(gains) + 0.008,
            f"n={len(data)}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#64748b",
        )

    ax.axhline(0, color="#64748b", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.set_ylabel("F1 gain: support-first - raw")
    ax.set_ylim(-0.01, 0.18)
    ax.set_xlim(-0.45, len(DATASET_ORDER) - 0.55)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, linestyle="-", alpha=0.95)
    fig.tight_layout()

    paths = save_figure(fig, outdir, "ipm_fig2_support_first_gain")
    copy_to_paper(paths, paper_fig_dir)
    return paths


def plot_no_context_gap(rows: list[dict[str, Any]], outdir: Path, paper_fig_dir: Path | None) -> list[Path]:
    by_key = {(row["dataset"], row["interface_group"]): row for row in rows}
    markers = {
        "no_context": "o",
        "raw_context": "o",
        "realistic_top5": "s",
        "raw_support_first": "D",
        "oracle_support": "^",
        "structured": "P",
    }
    offsets = {
        "no_context": -0.24,
        "realistic_top5": -0.12,
        "raw_context": 0.00,
        "raw_support_first": 0.12,
        "oracle_support": 0.24,
        "structured": 0.34,
    }

    fig, ax = plt.subplots(figsize=(5.05, 2.65))
    y_positions = list(reversed(range(len(DATASET_ORDER))))
    dataset_to_y = dict(zip(DATASET_ORDER, y_positions))

    for dataset in DATASET_ORDER:
        y_base = dataset_to_y[dataset]
        values = [
            by_key[(dataset, group)]["f1"]
            for group in N2000_INTERFACE_ORDER
            if (dataset, group) in by_key
        ]
        no_context = by_key.get((dataset, "no_context"))
        if no_context and values:
            ax.plot(
                [no_context["f1"], max(values)],
                [y_base, y_base],
                color=COLORS["range"],
                linewidth=2.2,
                solid_capstyle="round",
                zorder=1,
            )
        for group in N2000_INTERFACE_ORDER:
            row = by_key.get((dataset, group))
            if not row:
                continue
            ax.scatter(
                row["f1"],
                y_base + offsets[group],
                s=42 if group != "structured" else 56,
                color=COLORS[group],
                marker=markers[group],
                edgecolor="white",
                linewidth=0.55,
                zorder=3,
                label=N2000_INTERFACE_LABELS[group],
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.set_xlabel("Answer F1")
    ax.set_xlim(0.05, 1.03)
    ax.set_ylim(-0.55, len(DATASET_ORDER) - 0.45)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, linestyle="-", alpha=0.95)

    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ordered_labels = [
        N2000_INTERFACE_LABELS[group]
        for group in N2000_INTERFACE_ORDER
        if N2000_INTERFACE_LABELS[group] in unique
    ]
    ax.legend(
        [unique[label] for label in ordered_labels],
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    fig.tight_layout()

    paths = save_figure(fig, outdir, "ipm_fig3_no_context_gap")
    copy_to_paper(paths, paper_fig_dir)
    return paths


def compact_interface_label(group: str) -> str:
    return {
        "raw_context": "Raw",
        "raw_support_first": "Support-first",
        "realistic_top5": "Top-5",
        "oracle_support": "Oracle",
    }[group]


def centered_offsets(count: int, span: float) -> list[float]:
    if count <= 1:
        return [0.0]
    step = span / (count - 1)
    start = -span / 2
    return [start + idx * step for idx in range(count)]


def short_model_label(label: str) -> str:
    return (
        label.replace("Qwen2.5-Coder-", "Qwen")
        .replace("Llama-3.2-", "Llama")
        .replace("Llama-3.1-", "Llama")
        .replace("Mistral-", "Mistral")
        .replace("Gemma-2-", "Gemma")
    )


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = [outdir / f"{stem}.pdf", outdir / f"{stem}.png"]
    for path in paths:
        fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return paths


def copy_to_paper(paths: Iterable[Path], paper_fig_dir: Path | None) -> None:
    if paper_fig_dir is None:
        return
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, paper_fig_dir / path.name)


def write_main_markdown(path: Path, main_rows: list[dict[str, Any]], gain_rows: list[dict[str, Any]]) -> None:
    main_pivot = pivot_main_table(main_rows)
    lines = [
        "# IPM Consolidated Main Results",
        "",
        "Generated from `experiments/outputs/rank_sweep/*summary.json` by `experiments/consolidate_ipm_results.py`.",
        "",
        "## Qwen14B n=4000 Main Interface Hierarchy",
        "",
        "|Dataset|Raw context|Support first|Realistic top-5|Oracle support|Structured triples|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in main_pivot:
        lines.append(
            "|{dataset_label}|{raw_context}|{raw_support_first}|{realistic_top5}|{oracle_support}|{structured}|".format(
                dataset_label=row["dataset_label"],
                raw_context=format_cell(row["raw_context"]),
                raw_support_first=format_cell(row["raw_support_first"]),
                realistic_top5=format_cell(row["realistic_top5"]),
                oracle_support=format_cell(row["oracle_support"]),
                structured=format_cell(row["structured"]),
            )
        )
    lines.extend(
        [
            "",
            "## Support-First Gain over Raw Context",
            "",
            "|Dataset|Model|Train size|Raw F1|Support-first F1|Gain|",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in gain_rows:
        lines.append(
            "|{dataset}|{model}|{train_size}|{raw}|{support}|{gain}|".format(
                dataset=row["dataset_label"],
                model=row["model_label"],
                train_size=row["train_size"],
                raw=format_cell(row["raw_context_f1"]),
                support=format_cell(row["support_first_f1"]),
                gain=format_signed(row["support_gain_f1"]),
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def format_cell(value: Any) -> str:
    if value == "" or value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_signed(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.3f}"
    return format_cell(value)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(PLOT_STYLE)
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    paper_fig_dir = None if args.skip_paper_copy else args.paper_fig_dir

    common_fields = [
        "dataset",
        "dataset_label",
        "model_label",
        "model_id",
        "interface_name",
        "interface_label",
        "train_size",
        "rank",
        "updates",
        "exact_match",
        "f1",
        "answer_contained",
        "avg_prompt_tokens",
        "truncated_count",
        "source_summary",
    ]

    all_rows = collect_all_rank_sweep_rows()
    write_csv(table_dir / "ipm_all_rank_sweep_results.csv", all_rows, common_fields + ["error"])

    qwen14b_main = collect_qwen14b_n4000()
    write_csv(
        table_dir / "ipm_qwen14b_n4000_interface_hierarchy.csv",
        qwen14b_main,
        common_fields + ["interface_group", "interface_group_label", "f1_gain_over_raw"],
    )
    write_csv(
        table_dir / "ipm_qwen14b_n4000_interface_hierarchy_pivot.csv",
        pivot_main_table(qwen14b_main),
        ["dataset_label"]
        + [item for group in MAIN_INTERFACE_ORDER for item in (group, f"{group}_tokens")],
    )

    qwen14b_controls = collect_qwen14b_n2000_controls()
    write_csv(table_dir / "ipm_qwen14b_n2000_controls.csv", qwen14b_controls, common_fields)
    qwen14b_no_context_gap = collect_qwen14b_n2000_no_context_gap(qwen14b_controls)
    write_csv(
        table_dir / "ipm_qwen14b_n2000_no_context_gap.csv",
        qwen14b_no_context_gap,
        common_fields + ["interface_group", "interface_group_label"],
    )

    cross_model = collect_cross_model_n1000()
    write_csv(table_dir / "ipm_cross_model_and_small_reader_core.csv", cross_model, common_fields + ["note"])

    gain_rows = support_gain_rows(cross_model + qwen14b_main)
    write_csv(
        table_dir / "ipm_support_first_gain_over_raw.csv",
        gain_rows,
        [
            "dataset",
            "dataset_label",
            "model_label",
            "train_size",
            "raw_context_f1",
            "support_first_f1",
            "support_gain_f1",
            "raw_tokens",
            "support_tokens",
        ],
    )

    write_main_markdown(table_dir / "ipm_main_result_tables.md", qwen14b_main, gain_rows)

    figure_paths = []
    figure_paths.extend(plot_qwen14b_interface_hierarchy(qwen14b_main, figure_dir, paper_fig_dir))
    figure_paths.extend(plot_support_gain(gain_rows, figure_dir, paper_fig_dir))
    figure_paths.extend(plot_no_context_gap(qwen14b_no_context_gap, figure_dir, paper_fig_dir))

    manifest = {
        "tables": [
            "tables/ipm_all_rank_sweep_results.csv",
            "tables/ipm_qwen14b_n4000_interface_hierarchy.csv",
            "tables/ipm_qwen14b_n4000_interface_hierarchy_pivot.csv",
            "tables/ipm_qwen14b_n2000_controls.csv",
            "tables/ipm_qwen14b_n2000_no_context_gap.csv",
            "tables/ipm_cross_model_and_small_reader_core.csv",
            "tables/ipm_support_first_gain_over_raw.csv",
            "tables/ipm_main_result_tables.md",
        ],
        "figures": [str(path.relative_to(args.output_dir)) for path in figure_paths],
        "paper_fig_dir": None if paper_fig_dir is None else str(paper_fig_dir),
    }
    (args.output_dir / "ipm_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
