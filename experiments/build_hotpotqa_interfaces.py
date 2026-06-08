#!/usr/bin/env python
"""Export fixed HotpotQA evidence-interface JSONL files.

Each source HotpotQA example is expanded into three rows:

1. no_context
2. raw_context
3. gold_supporting_sentences

The output is intentionally plain JSONL so later training, evaluation, and
intrinsic-dimension scripts can all consume the same frozen interface files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence

from datasets import Dataset

from inspect_hotpotqa import (
    approx_tokens,
    context_docs,
    extract_gold_sentences,
    flatten_context,
    load_split,
    normalize_text,
    supporting_facts,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "hotpotqa_interfaces"
INTERFACE_NAMES = ("no_context", "raw_context", "gold_supporting_sentences")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HotpotQA evidence-interface JSONL.")
    parser.add_argument("--dataset-name", default="hotpotqa/hotpot_qa")
    parser.add_argument("--config", default="distractor")
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


def select_examples(ds: Dataset, size: int, seed: int, shuffle: bool) -> Dataset:
    ds = ds.add_column("_source_index", list(range(len(ds))))
    if shuffle:
        ds = ds.shuffle(seed=seed)
    n = len(ds) if size <= 0 else min(size, len(ds))
    return ds.select(range(n))


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


def build_rows(example: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    docs = context_docs(example)
    facts = supporting_facts(example)
    gold_sentences, missing = extract_gold_sentences(docs, facts)
    raw_context = flatten_context(docs)
    gold_context = " ".join(gold_sentences)

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

    evidence_by_interface = {
        "no_context": "",
        "raw_context": raw_context,
        "gold_supporting_sentences": gold_context,
    }

    rows: List[Dict[str, Any]] = []
    for interface_name in INTERFACE_NAMES:
        evidence = evidence_by_interface[interface_name]
        input_text = make_input_text(question, interface_name, evidence)
        metadata = dict(shared_metadata)
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


def export_split(
    split: str,
    size: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    ds = load_split(args.dataset_name, args.config, split)
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
        "source_examples": len(selected),
        "expanded_rows": combined_count,
        "combined_file": str(combined_path),
        "interface_files": interface_files,
        "stats_by_interface": stats_summary,
    }


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_manifest(manifest: Dict[str, Any]) -> None:
    print("HotpotQA interface export")
    print(f"- output_dir: {manifest['output_dir']}")
    print(f"- seed: {manifest['seed']}")
    for split_name, split_info in manifest["splits"].items():
        print(
            f"- {split_name}: "
            f"{split_info['source_examples']} source examples -> "
            f"{split_info['expanded_rows']} rows"
        )
        for interface_name in INTERFACE_NAMES:
            tokens = split_info["stats_by_interface"][interface_name]["input_tokens"]
            evidence = split_info["stats_by_interface"][interface_name]["interface_evidence_tokens"]
            print(
                f"  - {interface_name}: "
                f"input median={tokens['median']:.0f}, "
                f"evidence median={evidence['median']:.0f}"
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
        "run_name": args.run_name,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "interfaces": list(INTERFACE_NAMES),
        "splits": splits,
    }
    write_manifest(output_dir / "manifest.json", manifest)
    print_manifest(manifest)


if __name__ == "__main__":
    main()
