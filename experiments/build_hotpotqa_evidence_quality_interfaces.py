#!/usr/bin/env python
"""Add controlled evidence-quality HotpotQA interface files.

This script uses the same deterministic HotpotQA subset selection as
build_hotpotqa_interfaces.py, then writes additional per-interface JSONL files
into the existing export directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from datasets import Dataset

from build_hotpotqa_interfaces import make_input_text, select_examples
from inspect_hotpotqa import (
    approx_tokens,
    context_docs,
    extract_gold_sentences,
    flatten_context,
    load_split,
    normalize_text,
    supporting_facts,
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hotpotqa_interfaces" / "pilot"
INTERFACE_NAMES = (
    "gold_plus_1_distractor",
    "gold_plus_3_distractors",
    "raw_support_first",
    "raw_support_first_2docs",
    "raw_support_first_3docs",
    "raw_support_first_5docs",
    "raw_bm25_order",
    "raw_bm25_top3docs",
    "raw_bm25_top5docs",
    "raw_support_middle",
    "raw_support_shuffled",
    "raw_support_last",
)

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
    parser = argparse.ArgumentParser(description="Build controlled HotpotQA evidence-quality interfaces.")
    parser.add_argument("--dataset-name", default="hotpotqa/hotpot_qa")
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing derived interface files.")
    return parser.parse_args()


def paragraph_block(doc: Dict[str, Any]) -> str:
    return f"{doc['title']}: {' '.join(doc['sentences'])}"


def text_tokens(text: str) -> List[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def build_gold_plus_distractors(
    gold_context: str, distractor_docs: Sequence[Dict[str, Any]], count: int
) -> Tuple[str, List[str]]:
    selected = list(distractor_docs[:count])
    distractor_text = "\n".join(paragraph_block(doc) for doc in selected)
    if distractor_text:
        evidence = "\n\n".join(["Supporting evidence:", gold_context, "Distractor context:", distractor_text])
    else:
        evidence = gold_context
    return evidence, [str(doc["title"]) for doc in selected]


def reorder_support_first(
    docs: Sequence[Dict[str, Any]], support_titles: Sequence[str]
) -> Tuple[str, List[str]]:
    support_set = set(support_titles)
    support_docs = [doc for doc in docs if doc["title"] in support_set]
    other_docs = [doc for doc in docs if doc["title"] not in support_set]
    ordered_docs = support_docs + other_docs
    return flatten_context(ordered_docs), [str(doc["title"]) for doc in ordered_docs]


def reorder_support_first_window(
    docs: Sequence[Dict[str, Any]], support_titles: Sequence[str], window_doc_count: int
) -> Tuple[str, List[str], List[str]]:
    support_set = set(support_titles)
    support_docs = [doc for doc in docs if doc["title"] in support_set]
    other_docs = [doc for doc in docs if doc["title"] not in support_set]
    ordered_docs = support_docs + other_docs
    window_docs = ordered_docs[:window_doc_count]
    return (
        flatten_context(window_docs),
        [str(doc["title"]) for doc in ordered_docs],
        [str(doc["title"]) for doc in window_docs],
    )


def reorder_support_last(
    docs: Sequence[Dict[str, Any]], support_titles: Sequence[str]
) -> Tuple[str, List[str]]:
    support_set = set(support_titles)
    support_docs = [doc for doc in docs if doc["title"] in support_set]
    other_docs = [doc for doc in docs if doc["title"] not in support_set]
    ordered_docs = other_docs + support_docs
    return flatten_context(ordered_docs), [str(doc["title"]) for doc in ordered_docs]


def reorder_support_middle(
    docs: Sequence[Dict[str, Any]], support_titles: Sequence[str]
) -> Tuple[str, List[str]]:
    support_set = set(support_titles)
    support_docs = [doc for doc in docs if doc["title"] in support_set]
    other_docs = [doc for doc in docs if doc["title"] not in support_set]
    midpoint = len(other_docs) // 2
    ordered_docs = other_docs[:midpoint] + support_docs + other_docs[midpoint:]
    return flatten_context(ordered_docs), [str(doc["title"]) for doc in ordered_docs]


def reorder_support_shuffled(
    docs: Sequence[Dict[str, Any]],
    support_titles: Sequence[str],
    source_id: str,
    seed: int,
) -> Tuple[str, List[str], int]:
    support_set = set(support_titles)
    support_docs = [doc for doc in docs if doc["title"] in support_set]
    other_docs = [doc for doc in docs if doc["title"] not in support_set]
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
    insert_at = int.from_bytes(digest[:8], "big") % (len(other_docs) + 1)
    ordered_docs = other_docs[:insert_at] + support_docs + other_docs[insert_at:]
    return flatten_context(ordered_docs), [str(doc["title"]) for doc in ordered_docs], insert_at


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
        term_counts = defaultdict(int)
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
    ordered_docs = [docs[idx] for idx in order]
    ordered_scores = [scores[idx] for idx in order]
    return ordered_docs, ordered_scores


def reorder_bm25(docs: Sequence[Dict[str, Any]], question: str) -> Tuple[str, List[str], List[float]]:
    ordered_docs, ordered_scores = rank_bm25_docs(docs, question)
    return flatten_context(ordered_docs), [str(doc["title"]) for doc in ordered_docs], ordered_scores


def reorder_bm25_window(
    docs: Sequence[Dict[str, Any]], question: str, window_doc_count: int
) -> Tuple[str, List[str], List[str], List[float]]:
    ordered_docs, ordered_scores = rank_bm25_docs(docs, question)
    window_docs = ordered_docs[:window_doc_count]
    return (
        flatten_context(window_docs),
        [str(doc["title"]) for doc in ordered_docs],
        [str(doc["title"]) for doc in window_docs],
        ordered_scores,
    )


def build_rows(example: Dict[str, Any], split: str, seed: int) -> List[Dict[str, Any]]:
    docs = context_docs(example)
    facts = supporting_facts(example)
    gold_sentences, missing = extract_gold_sentences(docs, facts)
    raw_context = flatten_context(docs)
    gold_context = " ".join(gold_sentences)

    support_titles = []
    seen = set()
    for title, _sent_id in facts:
        if title not in seen:
            support_titles.append(title)
            seen.add(title)
    distractor_docs = [doc for doc in docs if doc["title"] not in seen]
    original_titles = [str(doc["title"]) for doc in docs]
    support_positions_original = [
        idx for idx, doc in enumerate(docs) if str(doc["title"]) in set(support_titles)
    ]

    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    source_id = str(example.get("id", "")).strip()
    source_index = int(example.get("_source_index", -1))
    answer_norm = normalize_text(answer)

    evidence_by_interface: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    evidence, titles = build_gold_plus_distractors(gold_context, distractor_docs, 1)
    evidence_by_interface["gold_plus_1_distractor"] = (
        evidence,
        {
            "distractor_count": len(titles),
            "selected_distractor_titles": titles,
            "support_titles": support_titles,
        },
    )
    evidence, titles = build_gold_plus_distractors(gold_context, distractor_docs, 3)
    evidence_by_interface["gold_plus_3_distractors"] = (
        evidence,
        {
            "distractor_count": len(titles),
            "selected_distractor_titles": titles,
            "support_titles": support_titles,
        },
    )
    evidence, ordered_titles = reorder_support_first(docs, support_titles)
    evidence_by_interface["raw_support_first"] = (
        evidence,
        {
            "support_titles": support_titles,
            "ordered_titles": ordered_titles,
            "original_titles": original_titles,
            "support_positions_original": support_positions_original,
            "support_doc_count": len(support_titles),
        },
    )
    for window_doc_count in (2, 3, 5):
        evidence, ordered_titles, window_titles = reorder_support_first_window(
            docs, support_titles, window_doc_count
        )
        evidence_by_interface[f"raw_support_first_{window_doc_count}docs"] = (
            evidence,
            {
                "support_titles": support_titles,
                "ordered_titles": ordered_titles,
                "window_titles": window_titles,
                "original_titles": original_titles,
                "support_positions_original": support_positions_original,
                "support_doc_count": len(support_titles),
                "window_doc_count": len(window_titles),
            },
        )
    evidence, ordered_titles, bm25_scores = reorder_bm25(docs, question)
    support_positions_bm25 = [
        idx for idx, title in enumerate(ordered_titles) if title in set(support_titles)
    ]
    evidence_by_interface["raw_bm25_order"] = (
        evidence,
        {
            "support_titles": support_titles,
            "ordered_titles": ordered_titles,
            "original_titles": original_titles,
            "support_positions_original": support_positions_original,
            "support_positions_bm25": support_positions_bm25,
            "support_doc_count": len(support_titles),
            "bm25_scores": bm25_scores,
        },
    )
    for window_doc_count in (3, 5):
        evidence, ordered_titles, window_titles, bm25_scores = reorder_bm25_window(
            docs, question, window_doc_count
        )
        support_positions_bm25 = [
            idx for idx, title in enumerate(ordered_titles) if title in set(support_titles)
        ]
        support_positions_window = [
            idx for idx, title in enumerate(window_titles) if title in set(support_titles)
        ]
        evidence_by_interface[f"raw_bm25_top{window_doc_count}docs"] = (
            evidence,
            {
                "support_titles": support_titles,
                "ordered_titles": ordered_titles,
                "window_titles": window_titles,
                "original_titles": original_titles,
                "support_positions_original": support_positions_original,
                "support_positions_bm25": support_positions_bm25,
                "support_positions_window": support_positions_window,
                "support_doc_count": len(support_titles),
                "matched_support_doc_count_window": len(support_positions_window),
                "window_doc_count": len(window_titles),
                "bm25_scores": bm25_scores,
            },
        )
    evidence, ordered_titles = reorder_support_middle(docs, support_titles)
    evidence_by_interface["raw_support_middle"] = (
        evidence,
        {
            "support_titles": support_titles,
            "ordered_titles": ordered_titles,
            "original_titles": original_titles,
            "support_positions_original": support_positions_original,
            "support_doc_count": len(support_titles),
        },
    )
    evidence, ordered_titles, insert_at = reorder_support_shuffled(docs, support_titles, source_id, seed)
    evidence_by_interface["raw_support_shuffled"] = (
        evidence,
        {
            "support_titles": support_titles,
            "ordered_titles": ordered_titles,
            "original_titles": original_titles,
            "support_positions_original": support_positions_original,
            "support_insert_index": insert_at,
            "support_doc_count": len(support_titles),
        },
    )
    evidence, ordered_titles = reorder_support_last(docs, support_titles)
    evidence_by_interface["raw_support_last"] = (
        evidence,
        {
            "support_titles": support_titles,
            "ordered_titles": ordered_titles,
            "original_titles": original_titles,
            "support_positions_original": support_positions_original,
            "support_doc_count": len(support_titles),
        },
    )

    shared_metadata = {
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "source_split": split,
        "source_index": source_index,
        "source_id": source_id,
        "type": example.get("type"),
        "level": example.get("level"),
        "answer_type": "yes_no" if answer_norm in {"yes", "no"} else "span",
        "doc_count": len(docs),
        "sentence_count": sum(len(doc["sentences"]) for doc in docs),
        "supporting_fact_count": len(facts),
        "matched_gold_sentence_count": len(gold_sentences),
        "missing_supporting_fact_count": len(missing),
        "raw_context_tokens": approx_tokens(raw_context),
        "gold_evidence_tokens": approx_tokens(gold_context),
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], force: bool) -> int:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_split(split: str, ds: Dataset, size: int, args: argparse.Namespace) -> Dict[str, Any]:
    selected = select_examples(ds, size, args.seed, shuffle=not args.no_shuffle)
    rows_by_interface: Dict[str, List[Dict[str, Any]]] = {name: [] for name in INTERFACE_NAMES}
    stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for example in selected:
        for row in build_rows(example, split, args.seed):
            interface_name = row["interface_name"]
            rows_by_interface[interface_name].append(row)
            metadata = row["metadata"]
            stats[interface_name]["input_tokens"].append(metadata["input_tokens"])
            stats[interface_name]["interface_evidence_tokens"].append(
                metadata["interface_evidence_tokens"]
            )
            stats[interface_name]["distractor_count"].append(metadata.get("distractor_count", 0))

    interface_files: Dict[str, str] = {}
    for interface_name, rows in rows_by_interface.items():
        path = args.data_dir / f"{split}__{interface_name}.jsonl"
        write_jsonl(path, rows, force=args.force)
        interface_files[interface_name] = str(path)

    return {
        "split": split,
        "source_examples": len(selected),
        "interface_files": interface_files,
        "stats_by_interface": {
            interface_name: {metric: summarize(values) for metric, values in metric_values.items()}
            for interface_name, metric_values in stats.items()
        },
    }


def write_manifest(args: argparse.Namespace, splits: Dict[str, Any]) -> Path:
    path = args.data_dir / "evidence_quality_manifest.json"
    if path.exists() and not args.force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    manifest = {
        "dataset_name": args.dataset_name,
        "config": args.config,
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "interfaces": list(INTERFACE_NAMES),
        "splits": splits,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def print_summary(splits: Dict[str, Any], manifest_path: Path) -> None:
    print("HotpotQA evidence-quality interface export")
    print(f"- manifest: {manifest_path}")
    for split_name, split_info in splits.items():
        print(f"- {split_name}: {split_info['source_examples']} source examples")
        for interface_name in INTERFACE_NAMES:
            stats = split_info["stats_by_interface"][interface_name]
            print(
                f"  - {interface_name}: "
                f"input median={stats['input_tokens']['median']:.0f}, "
                f"evidence median={stats['interface_evidence_tokens']['median']:.0f}, "
                f"distractors median={stats['distractor_count']['median']:.0f}"
            )


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    train = load_split(args.dataset_name, args.config, "train")
    validation = load_split(args.dataset_name, args.config, "validation")
    splits = {
        "train": export_split("train", train, args.train_size, args),
        "validation": export_split("validation", validation, args.validation_size, args),
    }
    manifest_path = write_manifest(args, splits)
    print_summary(splits, manifest_path)


if __name__ == "__main__":
    main()
