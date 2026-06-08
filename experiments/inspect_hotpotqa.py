#!/usr/bin/env python
"""Inspect HotpotQA evidence interfaces for the RAG evidence project.

The script builds lightweight dataset statistics for three interfaces:
question-only, raw context, and gold supporting sentences. It does not train
or call any model. The goal is to verify whether HotpotQA provides a clean
evidence-control surface before running intrinsic-dimension experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_DATASET = "hotpotqa/hotpot_qa"
DEFAULT_CONFIG = "distractor"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


ARTICLES = {"a", "an", "the"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect HotpotQA context and supporting-fact evidence interfaces."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--example-count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--summary-name",
        default="hotpotqa_inspection_summary.json",
        help="JSON filename for aggregate statistics.",
    )
    parser.add_argument(
        "--examples-name",
        default="hotpotqa_interface_examples.jsonl",
        help="JSONL filename for transformed example previews.",
    )
    return parser.parse_args()


def load_split(dataset_name: str, config: str | None, split: str) -> Dataset:
    """Load one split, with a small fallback for dataset mirrors."""
    attempts: List[Tuple[str, str | None, Dict[str, Any]]] = []
    if config:
        attempts.append((dataset_name, config, {}))
        attempts.append((dataset_name, config, {"trust_remote_code": True}))
    attempts.append((dataset_name, None, {}))
    attempts.append((dataset_name, None, {"trust_remote_code": True}))

    last_error: Exception | None = None
    for name, cfg, kwargs in attempts:
        try:
            if cfg:
                data = load_dataset(name, cfg, **kwargs)
            else:
                data = load_dataset(name, **kwargs)
            if isinstance(data, DatasetDict):
                if split not in data:
                    available = ", ".join(data.keys())
                    raise KeyError(f"split '{split}' not found; available: {available}")
                return data[split]
            if split not in ("train", "all"):
                raise KeyError(f"loaded a Dataset object, but requested split '{split}'")
            return data
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:  # Keep trying mirrors/kwargs before failing.
            last_error = exc
            continue

    raise RuntimeError(f"Could not load dataset '{dataset_name}' ({config}): {last_error}")


def normalize_text(text: str) -> str:
    text = text.lower().translate(PUNCT_TABLE)
    words = [word for word in text.split() if word not in ARTICLES]
    return " ".join(words)


def approx_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def contains_answer(text: str, answer: str) -> bool:
    answer_norm = normalize_text(str(answer))
    if not answer_norm or answer_norm in {"yes", "no", "noanswer"}:
        return False
    return answer_norm in normalize_text(text)


def context_docs(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return context as [{'title': str, 'sentences': [str, ...]}, ...]."""
    ctx = example.get("context", [])

    if isinstance(ctx, dict):
        titles = ctx.get("title", []) or []
        sentences = ctx.get("sentences", []) or []
        docs = []
        for title, sent_list in zip(titles, sentences):
            docs.append({"title": str(title), "sentences": [str(s) for s in sent_list]})
        return docs

    docs = []
    if isinstance(ctx, list):
        for item in ctx:
            if isinstance(item, dict):
                title = item.get("title", "")
                sent_list = item.get("sentences", [])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title, sent_list = item[0], item[1]
            else:
                continue
            docs.append({"title": str(title), "sentences": [str(s) for s in sent_list]})
    return docs


def supporting_facts(example: Dict[str, Any]) -> List[Tuple[str, int]]:
    facts = example.get("supporting_facts", [])
    pairs: List[Tuple[str, int]] = []

    if isinstance(facts, dict):
        titles = facts.get("title", []) or []
        sent_ids = facts.get("sent_id", []) or []
        for title, sent_id in zip(titles, sent_ids):
            pairs.append((str(title), int(sent_id)))
        return pairs

    if isinstance(facts, list):
        for item in facts:
            if isinstance(item, dict):
                title = item.get("title", "")
                sent_id = item.get("sent_id", item.get("sent_id".upper(), -1))
                pairs.append((str(title), int(sent_id)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), int(item[1])))
    return pairs


def flatten_context(docs: Sequence[Dict[str, Any]]) -> str:
    blocks = []
    for doc in docs:
        title = doc["title"]
        sent_text = " ".join(doc["sentences"])
        blocks.append(f"{title}: {sent_text}")
    return "\n".join(blocks)


def extract_gold_sentences(
    docs: Sequence[Dict[str, Any]], facts: Sequence[Tuple[str, int]]
) -> Tuple[List[str], List[Tuple[str, int]]]:
    by_title = {doc["title"]: doc["sentences"] for doc in docs}
    gold: List[str] = []
    missing: List[Tuple[str, int]] = []

    for title, sent_id in facts:
        sentences = by_title.get(title)
        if sentences is None or sent_id < 0 or sent_id >= len(sentences):
            missing.append((title, sent_id))
            continue
        gold.append(sentences[sent_id])
    return gold, missing


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[int(idx)])
    return float(ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo))


def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def build_interface_example(example: Dict[str, Any]) -> Dict[str, Any]:
    docs = context_docs(example)
    facts = supporting_facts(example)
    gold_sentences, missing = extract_gold_sentences(docs, facts)
    raw_context = flatten_context(docs)
    gold_context = " ".join(gold_sentences)

    return {
        "id": example.get("id"),
        "question": example.get("question"),
        "answer": example.get("answer"),
        "metadata": {
            "type": example.get("type"),
            "level": example.get("level"),
            "doc_count": len(docs),
            "sentence_count": sum(len(doc["sentences"]) for doc in docs),
            "supporting_fact_count": len(facts),
            "matched_gold_sentence_count": len(gold_sentences),
            "missing_supporting_facts": missing,
        },
        "interfaces": {
            "no_context": str(example.get("question", "")),
            "raw_context": raw_context,
            "gold_supporting_sentences": gold_context,
        },
    }


def inspect_dataset(ds: Dataset, max_samples: int, example_count: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = min(max_samples, len(ds)) if max_samples > 0 else len(ds)
    selected = ds.select(range(n))

    rows: List[Dict[str, Any]] = []
    previews: List[Dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()

    for idx, example in enumerate(selected):
        docs = context_docs(example)
        facts = supporting_facts(example)
        gold_sentences, missing = extract_gold_sentences(docs, facts)

        raw_context = flatten_context(docs)
        gold_context = " ".join(gold_sentences)
        support_titles = {title for title, _ in facts}
        all_titles = {doc["title"] for doc in docs}

        type_counts[str(example.get("type", ""))] += 1
        level_counts[str(example.get("level", ""))] += 1

        raw_tokens = approx_tokens(raw_context)
        gold_tokens = approx_tokens(gold_context)
        rows.append(
            {
                "doc_count": len(docs),
                "sentence_count": sum(len(doc["sentences"]) for doc in docs),
                "supporting_fact_count": len(facts),
                "supporting_title_count": len(support_titles),
                "matched_gold_sentence_count": len(gold_sentences),
                "missing_supporting_fact_count": len(missing),
                "distractor_doc_count": max(0, len(all_titles - support_titles)),
                "raw_chars": len(raw_context),
                "gold_chars": len(gold_context),
                "raw_tokens": raw_tokens,
                "gold_tokens": gold_tokens,
                "compression_ratio_gold_to_raw": (gold_tokens / raw_tokens) if raw_tokens else 0.0,
                "answer_in_raw_context": contains_answer(raw_context, str(example.get("answer", ""))),
                "answer_in_gold_sentences": contains_answer(gold_context, str(example.get("answer", ""))),
                "yes_no_answer": normalize_text(str(example.get("answer", ""))) in {"yes", "no"},
            }
        )

        if idx < example_count:
            preview = build_interface_example(example)
            preview["interfaces"] = {
                name: truncate(value, 1200) for name, value in preview["interfaces"].items()
            }
            previews.append(preview)

    def col(name: str) -> List[Any]:
        return [row[name] for row in rows]

    missing_total = sum(col("missing_supporting_fact_count"))
    supporting_total = sum(col("supporting_fact_count"))
    non_yes_no = [row for row in rows if not row["yes_no_answer"]]

    summary = {
        "dataset": {
            "num_rows_in_split": len(ds),
            "num_rows_inspected": n,
            "features": repr(ds.features),
            "type_counts": dict(type_counts),
            "level_counts": dict(level_counts),
        },
        "context_stats": {
            "doc_count": summarize_numeric(col("doc_count")),
            "sentence_count": summarize_numeric(col("sentence_count")),
            "distractor_doc_count": summarize_numeric(col("distractor_doc_count")),
            "raw_context_tokens": summarize_numeric(col("raw_tokens")),
            "raw_context_chars": summarize_numeric(col("raw_chars")),
        },
        "gold_evidence_stats": {
            "supporting_fact_count": summarize_numeric(col("supporting_fact_count")),
            "supporting_title_count": summarize_numeric(col("supporting_title_count")),
            "matched_gold_sentence_count": summarize_numeric(col("matched_gold_sentence_count")),
            "gold_sentence_tokens": summarize_numeric(col("gold_tokens")),
            "gold_sentence_chars": summarize_numeric(col("gold_chars")),
            "compression_ratio_gold_to_raw": summarize_numeric(col("compression_ratio_gold_to_raw")),
            "missing_supporting_fact_count": missing_total,
            "missing_supporting_fact_rate": (missing_total / supporting_total) if supporting_total else 0.0,
        },
        "answer_leakage": {
            "non_yes_no_examples": len(non_yes_no),
            "answer_in_raw_context_rate": (
                sum(row["answer_in_raw_context"] for row in non_yes_no) / len(non_yes_no)
                if non_yes_no
                else 0.0
            ),
            "answer_in_gold_sentences_rate": (
                sum(row["answer_in_gold_sentences"] for row in non_yes_no) / len(non_yes_no)
                if non_yes_no
                else 0.0
            ),
        },
        "interfaces": {
            "no_context": "question only",
            "raw_context": "question plus all HotpotQA context documents",
            "gold_supporting_sentences": "question plus matched supporting-fact sentences",
        },
    }
    return summary, previews


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 15].rstrip() + " ...[truncated]"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(summary: Dict[str, Any], output_dir: Path) -> None:
    dataset = summary["dataset"]
    context = summary["context_stats"]
    gold = summary["gold_evidence_stats"]
    leakage = summary["answer_leakage"]

    print("HotpotQA inspection summary")
    print(f"- inspected rows: {dataset['num_rows_inspected']} / {dataset['num_rows_in_split']}")
    print(f"- type counts: {dataset['type_counts']}")
    print(f"- level counts: {dataset['level_counts']}")
    print(
        "- context: "
        f"docs mean={context['doc_count']['mean']:.2f}, "
        f"sentences mean={context['sentence_count']['mean']:.2f}, "
        f"raw tokens median={context['raw_context_tokens']['median']:.0f}"
    )
    print(
        "- gold evidence: "
        f"support facts mean={gold['supporting_fact_count']['mean']:.2f}, "
        f"matched mean={gold['matched_gold_sentence_count']['mean']:.2f}, "
        f"gold tokens median={gold['gold_sentence_tokens']['median']:.0f}, "
        f"gold/raw token ratio mean={gold['compression_ratio_gold_to_raw']['mean']:.3f}"
    )
    print(
        "- support matching: "
        f"missing={gold['missing_supporting_fact_count']} "
        f"rate={gold['missing_supporting_fact_rate']:.4f}"
    )
    print(
        "- answer leakage, non-yes/no only: "
        f"raw={leakage['answer_in_raw_context_rate']:.3f}, "
        f"gold={leakage['answer_in_gold_sentences_rate']:.3f}, "
        f"n={leakage['non_yes_no_examples']}"
    )
    print(f"- wrote outputs under: {output_dir}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_split(args.dataset_name, args.config, args.split)
    summary, examples = inspect_dataset(ds, args.max_samples, args.example_count)

    summary_path = args.output_dir / args.summary_name
    examples_path = args.output_dir / args.examples_name
    write_json(summary_path, summary)
    write_jsonl(examples_path, examples)
    print_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()
