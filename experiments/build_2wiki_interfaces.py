#!/usr/bin/env python
"""Export 2WikiMultiHopQA evidence-interface JSONL files.

The output schema intentionally matches the HotpotQA interface exports so the
existing LoRA training, generation evaluation, and answer-NLL scripts can be
reused with only a different --data-dir and --interfaces list.
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
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

from huggingface_hub import hf_hub_download


DEFAULT_DATASET_REPO = "voidful/2WikiMultihopQA"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "2wiki_interfaces"
INTERFACE_NAMES = (
    "no_context",
    "raw_context",
    "gold_supporting_sentences",
    "gold_evidence_triples",
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
    parser = argparse.ArgumentParser(description="Export 2WikiMultiHopQA evidence interfaces.")
    parser.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=None,
        help="Directory containing train.json and dev.json. If omitted, files are downloaded from HF.",
    )
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="pilot")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Use the first N rows instead of a deterministic reservoir sample.",
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


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """Stream a top-level JSON array without loading the whole file."""

    decoder = json.JSONDecoder()
    buffer = ""
    cursor = 0

    def fill(handle) -> bool:
        nonlocal buffer
        chunk = handle.read(chunk_size)
        if not chunk:
            return False
        buffer += chunk
        return True

    with path.open(encoding="utf-8") as handle:
        while True:
            if cursor >= len(buffer) and not fill(handle):
                return
            while cursor < len(buffer) and buffer[cursor].isspace():
                cursor += 1
            if cursor < len(buffer):
                break

        if buffer[cursor] != "[":
            raise ValueError(f"Expected top-level JSON array in {path}")
        cursor += 1

        while True:
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor < len(buffer):
                    break
                if not fill(handle):
                    return

            if buffer[cursor] == "]":
                return
            if buffer[cursor] == ",":
                cursor += 1
                continue

            while True:
                try:
                    item, end = decoder.raw_decode(buffer, cursor)
                    break
                except json.JSONDecodeError:
                    if not fill(handle):
                        raise

            if not isinstance(item, dict):
                raise ValueError(f"Expected object item in {path}")
            yield item
            cursor = end

            if cursor > 4 * chunk_size:
                buffer = buffer[cursor:]
                cursor = 0


def resolve_raw_files(args: argparse.Namespace) -> Dict[str, Path]:
    if args.raw_data_dir is not None:
        return {
            "train": args.raw_data_dir / "train.json",
            "validation": args.raw_data_dir / "dev.json",
        }

    return {
        "train": Path(
            hf_hub_download(
                repo_id=args.dataset_repo,
                filename="train.json",
                repo_type="dataset",
            )
        ),
        "validation": Path(
            hf_hub_download(
                repo_id=args.dataset_repo,
                filename="dev.json",
                repo_type="dataset",
            )
        ),
    }


def select_examples(path: Path, size: int, seed: int, shuffle: bool) -> Tuple[List[Dict[str, Any]], int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw 2Wiki file: {path}")

    if size <= 0:
        selected: List[Dict[str, Any]] = []
        seen = 0
        for source_index, example in enumerate(iter_json_array(path)):
            example["_source_index"] = source_index
            selected.append(example)
            seen += 1
        return selected, seen

    if not shuffle:
        selected = []
        seen = 0
        for source_index, example in enumerate(iter_json_array(path)):
            seen += 1
            if len(selected) < size:
                example["_source_index"] = source_index
                selected.append(example)
            if len(selected) >= size:
                break
        return selected, seen

    rng = random.Random(seed)
    reservoir: List[Dict[str, Any]] = []
    seen = 0
    for source_index, example in enumerate(iter_json_array(path)):
        seen += 1
        example["_source_index"] = source_index
        if len(reservoir) < size:
            reservoir.append(example)
            continue
        replace_at = rng.randint(0, source_index)
        if replace_at < size:
            reservoir[replace_at] = example
    rng.shuffle(reservoir)
    return reservoir, seen


def context_docs(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for item in example.get("context", []) or []:
        if isinstance(item, dict):
            title = item.get("title", "")
            sentences = item.get("sentences", [])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sentences = item[0], item[1]
        else:
            continue
        docs.append({"title": str(title), "sentences": [str(sentence) for sentence in sentences]})
    return docs


def supporting_facts(example: Dict[str, Any]) -> List[Tuple[str, int]]:
    facts: List[Tuple[str, int]] = []
    for item in example.get("supporting_facts", []) or []:
        if isinstance(item, dict):
            title = item.get("title", "")
            sent_id = item.get("sent_id", -1)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sent_id = item[0], item[1]
        else:
            continue
        try:
            facts.append((str(title), int(sent_id)))
        except (TypeError, ValueError):
            facts.append((str(title), -1))
    return facts


def evidence_triples(example: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for item in example.get("evidences", []) or []:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            triples.append((str(item[0]), str(item[1]), str(item[2])))
        elif isinstance(item, dict):
            subject = item.get("subject", item.get("head", ""))
            relation = item.get("relation", item.get("predicate", ""))
            obj = item.get("object", item.get("tail", ""))
            triples.append((str(subject), str(relation), str(obj)))
    return triples


def paragraph_block(doc: Dict[str, Any]) -> str:
    return f"{doc['title']}: {' '.join(doc['sentences'])}"


def flatten_context(docs: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(paragraph_block(doc) for doc in docs)


def extract_gold_sentences(
    docs: Sequence[Dict[str, Any]], facts: Sequence[Tuple[str, int]]
) -> Tuple[List[str], List[Tuple[str, int]]]:
    by_title = {doc["title"]: doc["sentences"] for doc in docs}
    by_normalized_title = {normalize_text(doc["title"]): doc["sentences"] for doc in docs}
    gold: List[str] = []
    missing: List[Tuple[str, int]] = []

    for title, sent_id in facts:
        sentences = by_title.get(title)
        if sentences is None:
            sentences = by_normalized_title.get(normalize_text(title))
        if sentences is None or sent_id < 0 or sent_id >= len(sentences):
            missing.append((title, sent_id))
            continue
        gold.append(sentences[sent_id])
    return gold, missing


def format_triples(triples: Sequence[Tuple[str, str, str]]) -> str:
    return "\n".join(f"{subject} -- {relation} --> {obj}" for subject, relation, obj in triples)


def support_titles_in_order(docs: Sequence[Dict[str, Any]], facts: Sequence[Tuple[str, int]]) -> List[str]:
    support_set = {normalize_text(title) for title, _sent_id in facts}
    ordered: List[str] = []
    for doc in docs:
        if normalize_text(doc["title"]) in support_set and doc["title"] not in ordered:
            ordered.append(str(doc["title"]))
    return ordered


def reorder_support_first(
    docs: Sequence[Dict[str, Any]], support_titles: Sequence[str]
) -> Tuple[str, List[str]]:
    support_set = {normalize_text(title) for title in support_titles}
    support_docs = [doc for doc in docs if normalize_text(doc["title"]) in support_set]
    other_docs = [doc for doc in docs if normalize_text(doc["title"]) not in support_set]
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


def build_rows(example: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    docs = context_docs(example)
    facts = supporting_facts(example)
    triples = evidence_triples(example)
    gold_sentences, missing = extract_gold_sentences(docs, facts)
    support_titles = support_titles_in_order(docs, facts)

    raw_context = flatten_context(docs)
    gold_context = " ".join(gold_sentences)
    triple_context = format_triples(triples)
    support_first_context, support_first_titles = reorder_support_first(docs, support_titles)
    bm25_docs, bm25_scores = rank_bm25_docs(docs, str(example.get("question", "")))
    bm25_order_context = flatten_context(bm25_docs)
    bm25_top5_docs = bm25_docs[:5]
    bm25_top5_context = flatten_context(bm25_top5_docs)

    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    source_id = str(example.get("_id", example.get("id", ""))).strip()
    source_index = int(example.get("_source_index", -1))
    answer_norm = normalize_text(answer)

    original_titles = [str(doc["title"]) for doc in docs]
    support_position_original = [
        idx for idx, title in enumerate(original_titles) if normalize_text(title) in {normalize_text(t) for t in support_titles}
    ]
    bm25_ordered_titles = [str(doc["title"]) for doc in bm25_docs]

    shared_metadata = {
        "dataset": DEFAULT_DATASET_REPO,
        "source_split": split,
        "source_index": source_index,
        "source_id": source_id,
        "type": example.get("type"),
        "answer_type": "yes_no" if answer_norm in {"yes", "no"} else "span",
        "doc_count": len(docs),
        "sentence_count": sum(len(doc["sentences"]) for doc in docs),
        "supporting_fact_count": len(facts),
        "matched_gold_sentence_count": len(gold_sentences),
        "missing_supporting_fact_count": len(missing),
        "relation_triple_count": len(triples),
        "raw_context_tokens": approx_tokens(raw_context),
        "gold_evidence_tokens": approx_tokens(gold_context),
        "gold_triple_tokens": approx_tokens(triple_context),
        "support_titles": support_titles,
        "original_titles": original_titles,
        "support_positions_original": support_position_original,
    }

    evidence_by_interface = {
        "no_context": ("", {}),
        "raw_context": (raw_context, {}),
        "gold_supporting_sentences": (gold_context, {"missing_supporting_facts": missing}),
        "gold_evidence_triples": (triple_context, {"relation_triples": triples}),
        "raw_support_first": (
            support_first_context,
            {
                "ordered_titles": support_first_titles,
                "support_positions_interface": list(range(len(support_titles))),
            },
        ),
        "raw_bm25_order": (
            bm25_order_context,
            {
                "ordered_titles": bm25_ordered_titles,
                "bm25_scores": bm25_scores,
                "support_positions_bm25": [
                    idx
                    for idx, title in enumerate(bm25_ordered_titles)
                    if normalize_text(title) in {normalize_text(t) for t in support_titles}
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
                "support_doc_count_in_window": sum(
                    1 for doc in bm25_top5_docs if normalize_text(doc["title"]) in {normalize_text(t) for t in support_titles}
                ),
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
    stats[key]["gold_triple_tokens"].append(metadata["gold_triple_tokens"])
    if key == "raw_bm25_top5docs":
        stats[key]["support_doc_count_in_window"].append(metadata["support_doc_count_in_window"])


def export_split(
    split: str,
    size: int,
    raw_path: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    selected, seen = select_examples(raw_path, size, args.seed, shuffle=not args.no_shuffle)

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
        "raw_file": str(raw_path),
        "source_rows_seen": seen,
        "source_examples": len(selected),
        "expanded_rows": combined_count,
        "combined_file": str(combined_path),
        "interface_files": interface_files,
        "stats_by_interface": stats_summary,
    }


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_manifest(manifest: Dict[str, Any]) -> None:
    print("2WikiMultiHopQA interface export")
    print(f"- output_dir: {manifest['output_dir']}")
    print(f"- seed: {manifest['seed']}")
    for split_name, split_info in manifest["splits"].items():
        print(
            f"- {split_name}: "
            f"{split_info['source_examples']} sampled from {split_info['source_rows_seen']} "
            f"-> {split_info['expanded_rows']} rows"
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

    raw_files = resolve_raw_files(args)
    splits = {
        "train": export_split("train", args.train_size, raw_files["train"], args, output_dir),
        "validation": export_split(
            "validation", args.validation_size, raw_files["validation"], args, output_dir
        ),
    }
    manifest = {
        "dataset_name": args.dataset_repo,
        "source_files": {split: str(path) for split, path in raw_files.items()},
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
