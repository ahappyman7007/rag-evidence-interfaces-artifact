#!/usr/bin/env python
"""Build MuSiQue interfaces ordered by a cross-encoder reranker.

The reranker scores each question-paragraph pair independently and reorders the
raw MuSiQue paragraphs by score. It does not use support labels at inference
time; support labels are recorded only for diagnostics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from datasets import load_dataset

from build_musique_interfaces import (
    DEFAULT_CONFIG,
    DEFAULT_DATASET_NAME,
    DEFAULT_OUTPUT_ROOT,
    answer_type,
    approx_tokens,
    flatten_context,
    make_input_text,
    paragraph_block,
    paragraph_docs,
    select_examples,
)


DEFAULT_DATA_DIR = DEFAULT_OUTPUT_ROOT / "pilot"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-encoder MuSiQue retrieval interfaces.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    parser.add_argument("--interface-prefix", default="raw_crossenc")
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--reranker-batch-size", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Use the first N rows instead of a deterministic shuffled subset.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing interface files.")
    return parser.parse_args()


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def load_reranker(model_name: str, device: str, local_files_only: bool):
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(model_name, device=device, local_files_only=local_files_only)
    except TypeError:
        return CrossEncoder(model_name, device=device)


def ordered_docs_by_scores(
    docs: Sequence[Dict[str, Any]], scores: Sequence[float]
) -> Tuple[List[Dict[str, Any]], List[float]]:
    order = sorted(range(len(docs)), key=lambda idx: (-float(scores[idx]), idx))
    ordered_docs = [docs[idx] for idx in order]
    ordered_scores = [float(scores[idx]) for idx in order]
    return ordered_docs, ordered_scores


def prepare_examples(split: str, size: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    ds = load_dataset(args.dataset_name, args.config, split=split)
    selected = select_examples(ds, size, args.seed, shuffle=not args.no_shuffle)
    prepared: List[Dict[str, Any]] = []
    for example in selected:
        docs = paragraph_docs(example)
        raw_context = flatten_context(docs)
        question = str(example.get("question", "")).strip()
        answer = str(example.get("answer", "")).strip()
        source_id = str(example.get("id", "")).strip()
        source_index = int(example.get("_source_index", -1))
        question_decomposition = example.get("question_decomposition", []) or []
        support_docs = [doc for doc in docs if doc.get("is_supporting")]
        original_titles = [str(doc["title"]) for doc in docs]
        support_titles = [str(doc["title"]) for doc in support_docs]
        support_indices = [int(doc["idx"]) for doc in support_docs]
        support_positions_original = [idx for idx, doc in enumerate(docs) if doc.get("is_supporting")]
        shared_metadata = {
            "dataset": args.dataset_name,
            "config": args.config,
            "source_split": split,
            "source_index": source_index,
            "source_id": source_id,
            "answer_type": answer_type(answer),
            "answerable": bool(example.get("answerable", True)),
            "answer_aliases": example.get("answer_aliases", []) or [],
            "hop_count": len(question_decomposition),
            "doc_count": len(docs),
            "support_doc_count": len(support_docs),
            "raw_context_tokens": approx_tokens(raw_context),
            "gold_evidence_tokens": approx_tokens(flatten_context(support_docs)),
            "support_titles": support_titles,
            "support_indices": support_indices,
            "original_titles": original_titles,
            "support_positions_original": support_positions_original,
            "question_decomposition": question_decomposition,
        }
        prepared.append(
            {
                "split": split,
                "source_id": source_id,
                "source_index": source_index,
                "question": question,
                "answer": answer,
                "docs": docs,
                "doc_texts": [paragraph_block(doc) for doc in docs],
                "support_titles": support_titles,
                "support_indices": support_indices,
                "original_titles": original_titles,
                "support_positions_original": support_positions_original,
                "shared_metadata": shared_metadata,
            }
        )
    return prepared


def score_pairs(model, prepared: List[Dict[str, Any]], batch_size: int) -> List[List[float]]:
    pairs: List[Tuple[str, str]] = []
    offsets: List[Tuple[int, int]] = []
    for item in prepared:
        start = len(pairs)
        pairs.extend((item["question"], doc_text) for doc_text in item["doc_texts"])
        offsets.append((start, len(item["doc_texts"])))

    print(f"scoring {len(pairs)} question-paragraph pairs")
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    scores = [float(score) for score in scores]
    return [scores[start : start + count] for start, count in offsets]


def build_interface_rows(
    prepared: Dict[str, Any],
    scores: Sequence[float],
    reranker_model: str,
    interface_prefix: str,
) -> List[Dict[str, Any]]:
    docs = prepared["docs"]
    ordered_docs, ordered_scores = ordered_docs_by_scores(docs, scores)
    ordered_titles = [str(doc["title"]) for doc in ordered_docs]
    support_positions_crossenc = [
        idx for idx, doc in enumerate(ordered_docs) if doc.get("is_supporting")
    ]

    interface_names = (
        f"{interface_prefix}_order",
        f"{interface_prefix}_top3docs",
        f"{interface_prefix}_top5docs",
    )
    evidence_by_interface: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    evidence_by_interface[interface_names[0]] = (
        flatten_context(ordered_docs),
        {
            "ordered_titles": ordered_titles,
            "crossenc_scores": ordered_scores,
            "support_positions_crossenc": support_positions_crossenc,
        },
    )
    for window_doc_count, interface_name in [(3, interface_names[1]), (5, interface_names[2])]:
        window_docs = ordered_docs[:window_doc_count]
        window_titles = [str(doc["title"]) for doc in window_docs]
        support_positions_window = [
            idx for idx, doc in enumerate(window_docs) if doc.get("is_supporting")
        ]
        support_doc_count_in_window = len(support_positions_window)
        support_doc_count = prepared["shared_metadata"]["support_doc_count"]
        evidence_by_interface[interface_name] = (
            flatten_context(window_docs),
            {
                "ordered_titles": ordered_titles,
                "window_titles": window_titles,
                "window_doc_count": len(window_titles),
                "crossenc_scores": ordered_scores[:window_doc_count],
                "support_positions_crossenc": support_positions_crossenc,
                "support_positions_window": support_positions_window,
                "support_doc_count_in_window": support_doc_count_in_window,
                "matched_support_doc_count_window": support_doc_count_in_window,
                "all_support_in_window": support_doc_count_in_window == support_doc_count,
            },
        )

    rows: List[Dict[str, Any]] = []
    for interface_name in interface_names:
        evidence, extra_metadata = evidence_by_interface[interface_name]
        input_text = make_input_text(prepared["question"], interface_name, evidence)
        metadata = dict(prepared["shared_metadata"])
        metadata.update(extra_metadata)
        metadata.update(
            {
                "reranker_model": reranker_model,
                "interface_evidence_tokens": approx_tokens(evidence),
                "input_tokens": approx_tokens(input_text),
            }
        )
        rows.append(
            {
                "id": f"{prepared['source_id']}::{interface_name}",
                "source_id": prepared["source_id"],
                "source_split": prepared["split"],
                "source_index": prepared["source_index"],
                "interface_name": interface_name,
                "question": prepared["question"],
                "answer": prepared["answer"],
                "target_text": prepared["answer"],
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


def export_split(split: str, size: int, model, args: argparse.Namespace) -> Dict[str, Any]:
    prepared = prepare_examples(split, size, args)
    print(f"{split}: {len(prepared)} examples")
    scores_by_example = score_pairs(model, prepared, args.reranker_batch_size)

    interface_names = (
        f"{args.interface_prefix}_order",
        f"{args.interface_prefix}_top3docs",
        f"{args.interface_prefix}_top5docs",
    )
    rows_by_interface: Dict[str, List[Dict[str, Any]]] = {name: [] for name in interface_names}
    stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for item, scores in zip(prepared, scores_by_example):
        for row in build_interface_rows(item, scores, args.reranker_model, args.interface_prefix):
            interface_name = row["interface_name"]
            rows_by_interface[interface_name].append(row)
            metadata = row["metadata"]
            stats[interface_name]["input_tokens"].append(metadata["input_tokens"])
            stats[interface_name]["interface_evidence_tokens"].append(
                metadata["interface_evidence_tokens"]
            )
            stats[interface_name]["support_doc_count"].append(metadata["support_doc_count"])
            if "support_doc_count_in_window" in metadata:
                stats[interface_name]["support_doc_count_in_window"].append(
                    metadata["support_doc_count_in_window"]
                )
                stats[interface_name]["all_support_in_window"].append(
                    float(metadata["all_support_in_window"])
                )
            if metadata.get("support_positions_crossenc"):
                stats[interface_name]["latest_support_position_crossenc"].append(
                    max(metadata["support_positions_crossenc"])
                )

    interface_files: Dict[str, str] = {}
    for interface_name, rows in rows_by_interface.items():
        path = args.data_dir / f"{split}__{interface_name}.jsonl"
        write_jsonl(path, rows, force=args.force)
        interface_files[interface_name] = str(path)

    return {
        "split": split,
        "source_examples": len(prepared),
        "interface_files": interface_files,
        "stats_by_interface": {
            interface_name: {metric: summarize(values) for metric, values in metric_values.items()}
            for interface_name, metric_values in stats.items()
        },
    }


def write_manifest(args: argparse.Namespace, splits: Dict[str, Any]) -> Path:
    path = args.data_dir / "musique_cross_encoder_retrieval_manifest.json"
    if path.exists() and not args.force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    manifest = {
        "dataset_name": args.dataset_name,
        "config": args.config,
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "reranker_model": args.reranker_model,
        "interface_prefix": args.interface_prefix,
        "interfaces": [
            f"{args.interface_prefix}_order",
            f"{args.interface_prefix}_top3docs",
            f"{args.interface_prefix}_top5docs",
        ],
        "splits": splits,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def print_summary(splits: Dict[str, Any], manifest_path: Path) -> None:
    print("MuSiQue cross-encoder retrieval interface export")
    print(f"- manifest: {manifest_path}")
    for split_name, split_info in splits.items():
        print(f"- {split_name}: {split_info['source_examples']} source examples")
        for interface_name, stats in split_info["stats_by_interface"].items():
            input_tokens = stats.get("input_tokens", {})
            evidence_tokens = stats.get("interface_evidence_tokens", {})
            line = (
                f"  - {interface_name}: "
                f"input median={input_tokens.get('median', 0.0):.0f}, "
                f"evidence median={evidence_tokens.get('median', 0.0):.0f}"
            )
            if "support_doc_count_in_window" in stats:
                line += (
                    ", matched support docs/window mean="
                    f"{stats['support_doc_count_in_window']['mean']:.2f}"
                )
            if "all_support_in_window" in stats:
                line += f", all-support rate={stats['all_support_in_window']['mean']:.3f}"
            if "latest_support_position_crossenc" in stats:
                line += (
                    ", latest support pos median="
                    f"{stats['latest_support_position_crossenc']['median']:.0f}"
                )
            print(line)


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    model = load_reranker(args.reranker_model, args.reranker_device, args.local_files_only)
    splits = {
        "train": export_split("train", args.train_size, model, args),
        "validation": export_split("validation", args.validation_size, model, args),
    }
    manifest_path = write_manifest(args, splits)
    print_summary(splits, manifest_path)


if __name__ == "__main__":
    main()
