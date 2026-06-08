#!/usr/bin/env python
"""Run a small model-only smoke test on exported HotpotQA interfaces."""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional for frozen baselines.
    PeftModel = None


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hotpotqa_interfaces" / "pilot"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "smoke_test"
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
INTERFACE_NAMES = ("no_context", "raw_context", "gold_supporting_sentences")

ARTICLES = {"a", "an", "the"}
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HotpotQA interface smoke test.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--interfaces", nargs="+", default=list(INTERFACE_NAMES))
    parser.add_argument("--max-examples-per-interface", type=int, default=20)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    words = [word for word in text.split() if word not in ARTICLES]
    return " ".join(words)


def clean_prediction(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(answer|final answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    first_line = text.splitlines()[0].strip() if text else ""
    first_line = first_line.strip(" \t'\"`")
    return first_line


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_text(prediction) == normalize_text(answer))


def answer_contained(prediction: str, answer: str) -> float:
    answer_norm = normalize_text(answer)
    prediction_norm = normalize_text(prediction)
    if answer_norm in {"yes", "no"}:
        return float(answer_norm == prediction_norm)
    if not answer_norm:
        return 0.0
    return float(answer_norm in prediction_norm)


def f1_score(prediction: str, answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    if not pred_tokens and not answer_tokens:
        return 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    answer_counts = Counter(answer_tokens)
    overlap = sum((pred_counts & answer_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def load_rows(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_eval_rows(
    data_dir: Path, split: str, interfaces: Iterable[str], limit: int
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for interface_name in interfaces:
        path = data_dir / f"{split}__{interface_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing interface file: {path}")
        rows.extend(load_rows(path, limit))
    return rows


def load_model(model_id: str, adapter_path: Path | None, device: str, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"local_files_only": local_files_only}
    if device.startswith("cuda"):
        model_kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter_path is not None:
        if PeftModel is None:
            raise RuntimeError("peft is required when --adapter-path is provided")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def make_prompt(tokenizer, input_text: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": input_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return input_text


@torch.inference_mode()
def generate_answer(
    tokenizer,
    model,
    input_text: str,
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    use_chat_template: bool,
) -> Dict[str, Any]:
    prompt = make_prompt(tokenizer, input_text, use_chat_template)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_len = encoded["input_ids"].shape[-1]

    start = time.time()
    output = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.time() - start
    generated_ids = output[0, input_len:]
    raw_prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "raw_prediction": raw_prediction,
        "prediction": clean_prediction(raw_prediction),
        "prompt_tokens": int(input_len),
        "generated_tokens": int(generated_ids.shape[-1]),
        "latency_seconds": elapsed,
        "truncated": bool(input_len >= max_input_tokens),
    }


def summarize_predictions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["interface_name"]].append(row)

    summary: Dict[str, Any] = {}
    for interface_name, items in grouped.items():
        summary[interface_name] = {
            "n": len(items),
            "exact_match": mean(item["metrics"]["exact_match"] for item in items) if items else 0.0,
            "f1": mean(item["metrics"]["f1"] for item in items) if items else 0.0,
            "answer_contained": mean(item["metrics"]["answer_contained"] for item in items)
            if items
            else 0.0,
            "avg_prompt_tokens": mean(item["generation"]["prompt_tokens"] for item in items) if items else 0.0,
            "avg_latency_seconds": mean(item["generation"]["latency_seconds"] for item in items) if items else 0.0,
            "truncated_count": sum(item["generation"]["truncated"] for item in items),
        }
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(summary: Dict[str, Any], output_dir: Path) -> None:
    print("HotpotQA smoke test")
    print(f"- output_dir: {output_dir}")
    for interface_name in summary.get("interfaces", INTERFACE_NAMES):
        if interface_name not in summary["metrics_by_interface"]:
            continue
        metrics = summary["metrics_by_interface"][interface_name]
        print(
            f"- {interface_name}: "
            f"n={metrics['n']}, "
            f"EM={metrics['exact_match']:.3f}, "
            f"F1={metrics['f1']:.3f}, "
            f"contains={metrics['answer_contained']:.3f}, "
            f"prompt_tokens={metrics['avg_prompt_tokens']:.1f}, "
            f"truncated={metrics['truncated_count']}"
        )


def main() -> None:
    args = parse_args()
    run_name = args.run_name or args.model_id.replace("/", "__")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows(
        args.data_dir,
        args.split,
        args.interfaces,
        args.max_examples_per_interface,
    )
    tokenizer, model = load_model(args.model_id, args.adapter_path, args.device, args.local_files_only)

    predictions: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        generation = generate_answer(
            tokenizer,
            model,
            row["input_text"],
            args.device,
            args.max_input_tokens,
            args.max_new_tokens,
            use_chat_template=not args.no_chat_template,
        )
        prediction = generation["prediction"]
        answer = row["answer"]
        predictions.append(
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "interface_name": row["interface_name"],
                "question": row["question"],
                "answer": answer,
                "prediction": prediction,
                "raw_prediction": generation["raw_prediction"],
                "metrics": {
                    "exact_match": exact_match(prediction, answer),
                    "f1": f1_score(prediction, answer),
                    "answer_contained": answer_contained(prediction, answer),
                },
                "generation": generation,
                "metadata": row.get("metadata", {}),
            }
        )
        if idx % 10 == 0 or idx == len(rows):
            print(f"generated {idx}/{len(rows)}")

    metrics_by_interface = summarize_predictions(predictions)
    summary = {
        "model_id": args.model_id,
        "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
        "data_dir": str(args.data_dir),
        "split": args.split,
        "interfaces": args.interfaces,
        "max_examples_per_interface": args.max_examples_per_interface,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "metrics_by_interface": metrics_by_interface,
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    print_summary(summary, output_dir)


if __name__ == "__main__":
    main()
