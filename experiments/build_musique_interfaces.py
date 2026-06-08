#!/usr/bin/env python
"""Export MuSiQue evidence-interface JSONL files.

The output schema matches the HotpotQA and 2Wiki exports so the existing LoRA
training and generation evaluation scripts can consume the data with only a
different --data-dir and --interfaces list.

The MuSiQue question_decomposition field is kept in metadata only. We do not
use it as an input interface in this first export because it contains
intermediate answers and may leak the final answer.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from datasets import Dataset, load_dataset


DEFAULT_DATASET_NAME = "bdsaglam/musique"
DEFAULT_CONFIG = "answerable"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "musique_interfaces"
INTERFACE_NAMES = (
    "no_context",
    "raw_context",
    "gold_supporting_paragraphs",
    "raw_support_first",
    "raw_bm25_order",
    "raw_bm25_top5docs",
)

ARTICLES = {"a", "an", "the"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MuSiQue evidence interfaces.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Use the first N rows instead of a deterministic shuffled subset.",
    )
    parser.add_argument(
        "--separate-interface-files",
        action="store_true",
        help="Also write one file per split/interface.",
    )
    return parser.parse_args()


def approx_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def normalize_text(text: str) -> str:
    text = text.lower().translate(PUNCT_TABLE)
    words = [word for word in text.split() if word not in ARTICLES]
    return " ".join(words)


def make_input_text(question: str, interface_name: str, evidence: str) -> str:
    if interface_name == "no_context":
        return "\n".join(
            [
                "Answer the question.",
                "",
                f"Question: {question}",
                "",
                "Return only the answer.",
            ]
        )

    return "\n".join(
        [
            "Answer the question using the evidence.",
            "",
            f"Question: {question}",
            "",
            "Evidence:",
            evidence,
            "",
            "Return only the answer.",
        ]
    )


def select_examples(ds: Dataset, size: int, seed: int, shuffle: bool) -> Dataset:
    ds = ds.add_column("_source_index", list(range(len(ds))))
    if shuffle:
        ds = ds.shuffle(seed=seed)
    n = len(ds) if size <= 0 else min(size, len(ds))
    return ds.select(range(n))


def paragraph_docs(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for item in example.get("paragraphs", []) or []:
        if not isinstance(item, dict):
            continue
        idx = item.get("idx", len(docs))
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            idx_int = len(docs)
        docs.append(
            {
                "idx": idx_int,
                "title": str(item.get("title", "")),
                "paragraph_text": str(item.get("paragraph_text", "")),
                "is_supporting": bool(item.get("is_supporting", False)),
            }
        )
    return docs


def paragraph_block(doc: Dict[str, Any]) -> str:
    return f"{doc['title']}: {doc['paragraph_text']}"


def flatten_context(docs: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(paragraph_block(doc) for doc in docs)


def support_docs_in_order(docs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [doc for doc in docs if doc.get("is_supporting")]


def reorder_support_first(docs: Sequence[Dict[str, Any]]) -> Tuple[str, List[str]]:
    support_docs = [doc for doc in docs if doc.get("is_supporting")]
    other_docs = [doc for doc in docs if not doc.get("is_supporting")]
    ordered_docs = support_docs + other_docs
    return flatten_context(ordered_docs), [str(doc["title"]) for doc in ordered_docs]


def text_tokens(text: str) -> List[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def rank_bm25_docs(docs: Sequence[Dict[str, Any]], question: str) -> Tuple[List[Dict[str, Any]], List[float]]:
    query_terms = text_tokens(question)
    doc_terms = [text_tokens(paragraph_block(doc)) for doc in docs]
    doc_count = len(docs)
    avg_len = mean(len(tokens) for tokens in doc_terms) if doc_terms else 1.0
    document_frequency: Dict[str, int] = defaultdict(int)
    for tokens in doc_terms:
        for term in set(tokens):
            document_frequency[term] += 1

    k1 = 1.2
    b = 0.75
    scores: List[float] = []
    for tokens in doc_terms:
        term_counts: Dict[str, int] = defaultdict(int)
        for term in tokens:
            term_counts[term] += 1
        doc_len = max(len(tokens), 1)
        score = 0.0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * doc_len / max(avg_len, 1.0))
            score += idf * (tf * (k1 + 1.0)) / denom
        scores.append(score)

    order = sorted(range(len(docs)), key=lambda idx: (-scores[idx], idx))
    return [docs[idx] for idx in order], [scores[idx] for idx in order]


def answer_type(answer: str) -> str:
    answer_norm = normalize_text(answer)
    if answer_norm in {"yes", "no"}:
        return "yes_no"
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", answer_norm):
        return "number"
    return "span"


def build_rows(example: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    docs = paragraph_docs(example)
    support_docs = support_docs_in_order(docs)
    raw_context = flatten_context(docs)
    gold_context = flatten_context(support_docs)
    support_first_context, support_first_titles = reorder_support_first(docs)
    bm25_docs, bm25_scores = rank_bm25_docs(docs, str(example.get("question", "")))
    bm25_order_context = flatten_context(bm25_docs)
    bm25_top5_docs = bm25_docs[:5]
    bm25_top5_context = flatten_context(bm25_top5_docs)

    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    source_id = str(example.get("id", "")).strip()
    source_index = int(example.get("_source_index", -1))
    question_decomposition = example.get("question_decomposition", []) or []
    answer_aliases = example.get("answer_aliases", []) or []

    original_titles = [str(doc["title"]) for doc in docs]
    support_titles = [str(doc["title"]) for doc in support_docs]
    support_indices = [int(doc["idx"]) for doc in support_docs]
    support_positions_original = [idx for idx, doc in enumerate(docs) if doc.get("is_supporting")]
    bm25_ordered_titles = [str(doc["title"]) for doc in bm25_docs]
    bm25_top5_support_count = sum(1 for doc in bm25_top5_docs if doc.get("is_supporting"))

    shared_metadata = {
        "dataset": DEFAULT_DATASET_NAME,
        "config": DEFAULT_CONFIG,
        "source_split": split,
        "source_index": source_index,
        "source_id": source_id,
        "answer_type": answer_type(answer),
        "answerable": bool(example.get("answerable", True)),
        "answer_aliases": answer_aliases,
        "hop_count": len(question_decomposition),
        "doc_count": len(docs),
        "support_doc_count": len(support_docs),
        "raw_context_tokens": approx_tokens(raw_context),
        "gold_evidence_tokens": approx_tokens(gold_context),
        "support_titles": support_titles,
        "support_indices": support_indices,
        "original_titles": original_titles,
        "support_positions_original": support_positions_original,
        "question_decomposition": question_decomposition,
    }

    evidence_by_interface = {
        "no_context": ("", {}),
        "raw_context": (raw_context, {}),
        "gold_supporting_paragraphs": (
            gold_context,
            {"support_positions_interface": list(range(len(support_docs)))},
        ),
        "raw_support_first": (
            support_first_context,
            {
                "ordered_titles": support_first_titles,
                "support_positions_interface": list(range(len(support_docs))),
            },
        ),
        "raw_bm25_order": (
            bm25_order_context,
            {
                "ordered_titles": bm25_ordered_titles,
                "bm25_scores": bm25_scores,
                "support_positions_bm25": [
                    idx for idx, doc in enumerate(bm25_docs) if doc.get("is_supporting")
                ],
            },
        ),
        "raw_bm25_top5docs": (
            bm25_top5_context,
            {
                "ordered_titles": bm25_ordered_titles,
                "window_titles": [str(doc["title"]) for doc in bm25_top5_docs],
                "bm25_scores": bm25_scores[:5],
                "window_doc_count": len(bm25_top5_docs),
                "support_doc_count_in_window": bm25_top5_support_count,
                "all_support_in_window": bm25_top5_support_count == len(support_docs),
            },
        ),
    }

    rows: List[Dict[str, Any]] = []
    for interface_name in INTERFACE_NAMES:
        evidence, extra_metadata = evidence_by_interface[interface_name]
        input_text = make_input_text(question, interface_name, evidence)
        metadata = dict(shared_metadata)
        metadata.update(extra_metadata)
        metadata["interface_evidence_tokens"] = approx_tokens(evidence)
        metadata["input_tokens"] = approx_tokens(input_text)

        rows.append(
            {
                "id": f"{source_id}::{interface_name}",
                "source_id": source_id,
                "source_split": split,
                "source_index": source_index,
                "interface_name": interface_name,
                "question": question,
                "answer": answer,
                "target_text": answer,
                "input_text": input_text,
                "metadata": metadata,
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def update_stats(stats: Dict[str, Dict[str, List[float]]], row: Dict[str, Any]) -> None:
    key = row["interface_name"]
    metadata = row["metadata"]
    stats[key]["input_tokens"].append(metadata["input_tokens"])
    stats[key]["interface_evidence_tokens"].append(metadata["interface_evidence_tokens"])
    stats[key]["raw_context_tokens"].append(metadata["raw_context_tokens"])
    stats[key]["gold_evidence_tokens"].append(metadata["gold_evidence_tokens"])
    stats[key]["support_doc_count"].append(metadata["support_doc_count"])
    if key == "raw_bm25_top5docs":
        stats[key]["support_doc_count_in_window"].append(metadata["support_doc_count_in_window"])
        stats[key]["all_support_in_window"].append(float(metadata["all_support_in_window"]))


def export_split(
    split: str,
    size: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    ds = load_dataset(args.dataset_name, args.config, split=split)
    selected = select_examples(ds, size, args.seed, shuffle=not args.no_shuffle)

    rows_by_interface: Dict[str, List[Dict[str, Any]]] = {name: [] for name in INTERFACE_NAMES}
    combined_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for example in selected:
        rows = build_rows(example, split)
        for row in rows:
            combined_rows.append(row)
            rows_by_interface[row["interface_name"]].append(row)
            update_stats(stats, row)

    combined_path = output_dir / f"{split}.jsonl"
    combined_count = write_jsonl(combined_path, combined_rows)

    interface_files: Dict[str, str] = {}
    if args.separate_interface_files:
        for interface_name, rows in rows_by_interface.items():
            path = output_dir / f"{split}__{interface_name}.jsonl"
            write_jsonl(path, rows)
            interface_files[interface_name] = str(path)

    stats_summary = {
        interface_name: {metric: summarize(values) for metric, values in metric_values.items()}
        for interface_name, metric_values in stats.items()
    }

    return {
        "split": split,
        "source_examples_total": len(ds),
        "source_examples": len(selected),
        "expanded_rows": combined_count,
        "combined_file": str(combined_path),
        "interface_files": interface_files,
        "stats_by_interface": stats_summary,
    }


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_manifest(manifest: Dict[str, Any]) -> None:
    print("MuSiQue interface export")
    print(f"- output_dir: {manifest['output_dir']}")
    print(f"- dataset: {manifest['dataset_name']} / {manifest['config']}")
    print(f"- seed: {manifest['seed']}")
    for split_name, split_info in manifest["splits"].items():
        print(
            f"- {split_name}: "
            f"{split_info['source_examples']} sampled from {split_info['source_examples_total']} "
            f"-> {split_info['expanded_rows']} rows"
        )
        for interface_name in INTERFACE_NAMES:
            stats = split_info["stats_by_interface"][interface_name]
            tokens = stats["input_tokens"]
            evidence = stats["interface_evidence_tokens"]
            extra = ""
            if interface_name == "raw_bm25_top5docs":
                coverage = stats["all_support_in_window"]["mean"]
                extra = f", all-support@5={coverage:.3f}"
            print(
                f"  - {interface_name}: "
                f"input median={tokens['median']:.0f}, "
                f"evidence median={evidence['median']:.0f}"
                f"{extra}"
            )


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": export_split("train", args.train_size, args, output_dir),
        "validation": export_split("validation", args.validation_size, args, output_dir),
    }

    manifest = {
        "dataset_name": args.dataset_name,
        "config": args.config,
        "seed": args.seed,
        "run_name": args.run_name,
        "output_dir": str(output_dir),
        "interfaces": list(INTERFACE_NAMES),
        "splits": splits,
        "notes": (
            "question_decomposition is stored in metadata only because it contains "
            "intermediate answers and may leak the final answer."
        ),
    }
    write_manifest(output_dir / "manifest.json", manifest)
    print_manifest(manifest)


if __name__ == "__main__":
    main()
