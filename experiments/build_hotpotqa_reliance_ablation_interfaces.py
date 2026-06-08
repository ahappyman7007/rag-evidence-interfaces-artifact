#!/usr/bin/env python
"""Build HotpotQA no-support ablation interfaces for evidence-reliance checks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from datasets import Dataset

from build_hotpotqa_evidence_quality_interfaces import reorder_support_first
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
BASE_INTERFACES = (
    "gold_supporting_sentences",
    "raw_context",
    "raw_support_first",
    "raw_ftcrossenc_top5docs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HotpotQA evidence-reliance ablation interfaces.")
    parser.add_argument("--dataset-name", default="hotpotqa/hotpot_qa")
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--interfaces", nargs="+", default=list(BASE_INTERFACES))
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def paragraph_block(doc: Dict[str, Any]) -> str:
    return f"{doc['title']}: {' '.join(doc['sentences'])}"


def support_titles_from_facts(facts: Sequence[Tuple[str, int]]) -> List[str]:
    titles: List[str] = []
    seen = set()
    for title, _sent_id in facts:
        if title not in seen:
            titles.append(str(title))
            seen.add(title)
    return titles


def docs_without_support_docs(
    docs: Sequence[Dict[str, Any]],
    support_titles: Sequence[str],
) -> List[Dict[str, Any]]:
    support_set = set(support_titles)
    return [doc for doc in docs if str(doc["title"]) not in support_set]


def docs_without_support_sentences(
    docs: Sequence[Dict[str, Any]],
    facts: Sequence[Tuple[str, int]],
) -> List[Dict[str, Any]]:
    fact_map: Dict[str, set[int]] = defaultdict(set)
    for title, sent_id in facts:
        fact_map[str(title)].add(int(sent_id))

    ablated: List[Dict[str, Any]] = []
    for doc in docs:
        title = str(doc["title"])
        remove_ids = fact_map.get(title, set())
        sentences = [
            sentence
            for idx, sentence in enumerate(doc["sentences"])
            if idx not in remove_ids
        ]
        ablated.append({"title": title, "sentences": sentences})
    return ablated


def rows_by_source(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["source_id"]] = row
    return rows


def ordered_docs_from_titles(
    docs: Sequence[Dict[str, Any]],
    titles: Sequence[str],
) -> List[Dict[str, Any]]:
    by_title = {str(doc["title"]): doc for doc in docs}
    ordered = []
    for title in titles:
        doc = by_title.get(str(title))
        if doc is not None:
            ordered.append(doc)
    return ordered


def summarize(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def build_evidence_for_interface(
    interface_name: str,
    docs: Sequence[Dict[str, Any]],
    facts: Sequence[Tuple[str, int]],
    support_titles: Sequence[str],
    existing_rows: Dict[str, Dict[str, Dict[str, Any]]],
    source_id: str,
) -> Tuple[str, Dict[str, Any]]:
    no_support_interface = f"{interface_name}_no_support_docs"

    if interface_name == "gold_supporting_sentences":
        return "", {
            "base_interface": interface_name,
            "ablation_type": "remove_all_gold_supporting_sentences",
            "support_titles": list(support_titles),
            "removed_support_doc_count": len(support_titles),
        }

    if interface_name == "raw_context":
        base_docs = docs
    elif interface_name == "raw_support_first":
        _evidence, ordered_titles = reorder_support_first(docs, support_titles)
        base_docs = ordered_docs_from_titles(docs, ordered_titles)
    elif interface_name == "raw_ftcrossenc_top5docs":
        existing = existing_rows.get(interface_name, {}).get(source_id)
        if existing is None:
            raise KeyError(f"Missing existing row for {source_id}::{interface_name}")
        window_titles = existing.get("metadata", {}).get("window_titles", [])
        base_docs = ordered_docs_from_titles(docs, window_titles)
    else:
        raise ValueError(f"Unsupported interface for ablation: {interface_name}")

    ablated_docs = docs_without_support_docs(base_docs, support_titles)
    removed_titles = [str(doc["title"]) for doc in base_docs if str(doc["title"]) in set(support_titles)]
    evidence = flatten_context(ablated_docs)
    return evidence, {
        "base_interface": interface_name,
        "ablation_type": "remove_support_docs",
        "ablation_interface": no_support_interface,
        "support_titles": list(support_titles),
        "removed_support_titles": removed_titles,
        "removed_support_doc_count": len(removed_titles),
        "remaining_doc_titles": [str(doc["title"]) for doc in ablated_docs],
        "remaining_doc_count": len(ablated_docs),
    }


def build_rows(
    example: Dict[str, Any],
    split: str,
    interfaces: Sequence[str],
    existing_rows: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    docs = context_docs(example)
    facts = supporting_facts(example)
    gold_sentences, missing = extract_gold_sentences(docs, facts)
    raw_context = flatten_context(docs)
    gold_context = " ".join(gold_sentences)
    support_titles = support_titles_from_facts(facts)

    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    source_id = str(example.get("id", "")).strip()
    source_index = int(example.get("_source_index", -1))
    answer_norm = normalize_text(answer)

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
    for interface_name in interfaces:
        evidence, extra_metadata = build_evidence_for_interface(
            interface_name,
            docs,
            facts,
            support_titles,
            existing_rows,
            source_id,
        )
        ablation_name = f"{interface_name}_no_support_docs"
        input_text = make_input_text(question, ablation_name, evidence)
        metadata = dict(shared_metadata)
        metadata.update(extra_metadata)
        metadata["interface_evidence_tokens"] = approx_tokens(evidence)
        metadata["input_tokens"] = approx_tokens(input_text)
        rows.append(
            {
                "id": f"{source_id}::{ablation_name}",
                "source_id": source_id,
                "source_split": split,
                "source_index": source_index,
                "interface_name": ablation_name,
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
    existing_rows = {
        interface_name: rows_by_source(args.data_dir / f"{split}__{interface_name}.jsonl")
        for interface_name in args.interfaces
    }
    rows_by_interface: Dict[str, List[Dict[str, Any]]] = {
        f"{interface_name}_no_support_docs": [] for interface_name in args.interfaces
    }
    stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for example in selected:
        for row in build_rows(example, split, args.interfaces, existing_rows):
            interface_name = row["interface_name"]
            rows_by_interface[interface_name].append(row)
            metadata = row["metadata"]
            stats[interface_name]["input_tokens"].append(metadata["input_tokens"])
            stats[interface_name]["interface_evidence_tokens"].append(
                metadata["interface_evidence_tokens"]
            )
            stats[interface_name]["removed_support_doc_count"].append(
                metadata["removed_support_doc_count"]
            )

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
    path = args.data_dir / "reliance_ablation_manifest.json"
    if path.exists() and not args.force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite")
    manifest = {
        "dataset_name": args.dataset_name,
        "config": args.config,
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "base_interfaces": list(args.interfaces),
        "ablation": "remove_support_docs",
        "splits": splits,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def print_summary(splits: Dict[str, Any], manifest_path: Path) -> None:
    print("HotpotQA reliance ablation interface export")
    print(f"- manifest: {manifest_path}")
    for split_name, split_info in splits.items():
        print(f"- {split_name}: {split_info['source_examples']} source examples")
        for interface_name, stats in split_info["stats_by_interface"].items():
            print(
                f"  - {interface_name}: "
                f"input median={stats['input_tokens']['median']:.0f}, "
                f"evidence median={stats['interface_evidence_tokens']['median']:.0f}, "
                f"removed support docs median={stats['removed_support_doc_count']['median']:.0f}"
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
