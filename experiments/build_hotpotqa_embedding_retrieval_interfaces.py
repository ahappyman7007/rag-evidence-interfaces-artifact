#!/usr/bin/env python
"""Build HotpotQA interfaces ordered by sentence-embedding retrieval.

This script adds realistic non-oracle retrieval interfaces without changing the
existing BM25/evidence-quality builder. It uses only the question and raw
document text to rank the 10 context documents.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from datasets import Dataset

from build_hotpotqa_interfaces import make_input_text, select_examples
from inspect_hotpotqa import (
    approx_tokens,
    context_docs,
    flatten_context,
    load_split,
    normalize_text,
    supporting_facts,
)


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hotpotqa_interfaces" / "pilot"
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
INTERFACE_NAMES = (
    "raw_embed_order",
    "raw_embed_top3docs",
    "raw_embed_top5docs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build embedding-retrieval HotpotQA interfaces.")
    parser.add_argument("--dataset-name", default="hotpotqa/hotpot_qa")
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--encoder-model", default=DEFAULT_ENCODER)
    parser.add_argument("--encoder-device", default="cpu")
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing interface files.")
    return parser.parse_args()


def paragraph_block(doc: Dict[str, Any]) -> str:
    return f"{doc['title']}: {' '.join(doc['sentences'])}"


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def load_encoder(model_name: str, device: str, local_files_only: bool):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
    except TypeError:
        return SentenceTransformer(model_name, device=device)


def encode_texts(model, texts: List[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def ordered_docs_by_scores(
    docs: Sequence[Dict[str, Any]], scores: np.ndarray
) -> Tuple[List[Dict[str, Any]], List[float]]:
    order = sorted(range(len(docs)), key=lambda idx: (-float(scores[idx]), idx))
    ordered_docs = [docs[idx] for idx in order]
    ordered_scores = [float(scores[idx]) for idx in order]
    return ordered_docs, ordered_scores


def support_titles_from_facts(facts: Sequence[Tuple[str, int]]) -> List[str]:
    support_titles: List[str] = []
    seen = set()
    for title, _sent_id in facts:
        if title not in seen:
            support_titles.append(str(title))
            seen.add(title)
    return support_titles


def build_interface_rows(
    prepared: Dict[str, Any],
    q_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    encoder_model: str,
) -> List[Dict[str, Any]]:
    docs = prepared["docs"]
    scores = doc_embeddings @ q_embedding
    ordered_docs, ordered_scores = ordered_docs_by_scores(docs, scores)
    ordered_titles = [str(doc["title"]) for doc in ordered_docs]

    support_titles = prepared["support_titles"]
    support_set = set(support_titles)
    support_positions_embed = [
        idx for idx, title in enumerate(ordered_titles) if title in support_set
    ]

    evidence_by_interface: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    evidence_by_interface["raw_embed_order"] = (
        flatten_context(ordered_docs),
        {
            "support_positions_embed": support_positions_embed,
            "ordered_titles": ordered_titles,
            "embed_scores": ordered_scores,
        },
    )
    for window_doc_count in (3, 5):
        window_docs = ordered_docs[:window_doc_count]
        window_titles = [str(doc["title"]) for doc in window_docs]
        support_positions_window = [
            idx for idx, title in enumerate(window_titles) if title in support_set
        ]
        evidence_by_interface[f"raw_embed_top{window_doc_count}docs"] = (
            flatten_context(window_docs),
            {
                "support_positions_embed": support_positions_embed,
                "support_positions_window": support_positions_window,
                "matched_support_doc_count_window": len(support_positions_window),
                "ordered_titles": ordered_titles,
                "window_titles": window_titles,
                "window_doc_count": len(window_titles),
                "embed_scores": ordered_scores,
            },
        )

    rows: List[Dict[str, Any]] = []
    for interface_name in INTERFACE_NAMES:
        evidence, extra_metadata = evidence_by_interface[interface_name]
        input_text = make_input_text(prepared["question"], interface_name, evidence)
        metadata = dict(prepared["shared_metadata"])
        metadata.update(extra_metadata)
        metadata.update(
            {
                "encoder_model": encoder_model,
                "support_titles": support_titles,
                "original_titles": prepared["original_titles"],
                "support_positions_original": prepared["support_positions_original"],
                "support_doc_count": len(support_titles),
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


def prepare_examples(split: str, ds: Dataset, size: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = select_examples(ds, size, args.seed, shuffle=not args.no_shuffle)
    prepared: List[Dict[str, Any]] = []
    for example in selected:
        docs = context_docs(example)
        facts = supporting_facts(example)
        support_titles = support_titles_from_facts(facts)
        original_titles = [str(doc["title"]) for doc in docs]
        support_set = set(support_titles)
        support_positions_original = [
            idx for idx, title in enumerate(original_titles) if title in support_set
        ]
        raw_context = flatten_context(docs)
        question = str(example.get("question", "")).strip()
        answer = str(example.get("answer", "")).strip()
        answer_norm = normalize_text(answer)
        source_id = str(example.get("id", "")).strip()
        source_index = int(example.get("_source_index", -1))
        shared_metadata = {
            "dataset": args.dataset_name,
            "config": args.config,
            "source_split": split,
            "source_index": source_index,
            "source_id": source_id,
            "type": example.get("type"),
            "level": example.get("level"),
            "answer_type": "yes_no" if answer_norm in {"yes", "no"} else "span",
            "doc_count": len(docs),
            "sentence_count": sum(len(doc["sentences"]) for doc in docs),
            "supporting_fact_count": len(facts),
            "raw_context_tokens": approx_tokens(raw_context),
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
                "original_titles": original_titles,
                "support_positions_original": support_positions_original,
                "shared_metadata": shared_metadata,
            }
        )
    return prepared


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], force: bool) -> int:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_split(split: str, ds: Dataset, size: int, model, args: argparse.Namespace) -> Dict[str, Any]:
    prepared = prepare_examples(split, ds, size, args)
    question_texts = [item["question"] for item in prepared]
    doc_texts: List[str] = []
    doc_offsets: List[Tuple[int, int]] = []
    for item in prepared:
        start = len(doc_texts)
        doc_texts.extend(item["doc_texts"])
        doc_offsets.append((start, len(item["doc_texts"])))

    print(f"encoding {split}: {len(question_texts)} questions, {len(doc_texts)} documents")
    q_embeddings = encode_texts(model, question_texts, args.encoder_batch_size)
    d_embeddings = encode_texts(model, doc_texts, args.encoder_batch_size)

    rows_by_interface: Dict[str, List[Dict[str, Any]]] = {name: [] for name in INTERFACE_NAMES}
    stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for idx, item in enumerate(prepared):
        start, count = doc_offsets[idx]
        doc_embedding_slice = d_embeddings[start : start + count]
        for row in build_interface_rows(
            item,
            q_embeddings[idx],
            doc_embedding_slice,
            args.encoder_model,
        ):
            interface_name = row["interface_name"]
            rows_by_interface[interface_name].append(row)
            metadata = row["metadata"]
            stats[interface_name]["input_tokens"].append(metadata["input_tokens"])
            stats[interface_name]["interface_evidence_tokens"].append(
                metadata["interface_evidence_tokens"]
            )
            if "matched_support_doc_count_window" in metadata:
                stats[interface_name]["matched_support_doc_count_window"].append(
                    metadata["matched_support_doc_count_window"]
                )
            if "support_positions_embed" in metadata and metadata["support_positions_embed"]:
                stats[interface_name]["latest_support_position_embed"].append(
                    max(metadata["support_positions_embed"])
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
    path = args.data_dir / "embedding_retrieval_manifest.json"
    if path.exists() and not args.force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    manifest = {
        "dataset_name": args.dataset_name,
        "config": args.config,
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "encoder_model": args.encoder_model,
        "interfaces": list(INTERFACE_NAMES),
        "splits": splits,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def print_summary(splits: Dict[str, Any], manifest_path: Path) -> None:
    print("HotpotQA embedding-retrieval interface export")
    print(f"- manifest: {manifest_path}")
    for split_name, split_info in splits.items():
        print(f"- {split_name}: {split_info['source_examples']} source examples")
        for interface_name in INTERFACE_NAMES:
            stats = split_info["stats_by_interface"][interface_name]
            input_tokens = stats.get("input_tokens", {})
            evidence_tokens = stats.get("interface_evidence_tokens", {})
            line = (
                f"  - {interface_name}: "
                f"input median={input_tokens.get('median', 0.0):.0f}, "
                f"evidence median={evidence_tokens.get('median', 0.0):.0f}"
            )
            if "matched_support_doc_count_window" in stats:
                line += (
                    ", matched support docs/window mean="
                    f"{stats['matched_support_doc_count_window']['mean']:.2f}"
                )
            if "latest_support_position_embed" in stats:
                line += (
                    ", latest support pos median="
                    f"{stats['latest_support_position_embed']['median']:.0f}"
                )
            print(line)


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    model = load_encoder(args.encoder_model, args.encoder_device, args.local_files_only)
    train = load_split(args.dataset_name, args.config, "train")
    validation = load_split(args.dataset_name, args.config, "validation")
    splits = {
        "train": export_split("train", train, args.train_size, model, args),
        "validation": export_split("validation", validation, args.validation_size, model, args),
    }
    manifest_path = write_manifest(args, splits)
    print_summary(splits, manifest_path)


if __name__ == "__main__":
    main()
