#!/usr/bin/env python
"""Compute answer negative log-likelihood for HotpotQA evidence interfaces."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional for frozen baselines.
    PeftModel = None


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hotpotqa_interfaces" / "pilot"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "answer_nll"
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute HotpotQA answer NLL.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--interfaces", nargs="+", required=True)
    parser.add_argument("--max-examples-per-interface", type=int, default=300)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def load_rows(data_dir: Path, split: str, interfaces: Iterable[str], limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for interface_name in interfaces:
        path = data_dir / f"{split}__{interface_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path, limit))
    return rows


def load_model(model_id: str, adapter_path: Path | None, device: str, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
    return input_text.rstrip() + "\nAnswer:"


def encode_row(tokenizer, row: Dict[str, Any], max_length: int, use_chat_template: bool) -> Dict[str, Any]:
    prompt_text = make_prompt(tokenizer, row["input_text"], use_chat_template)
    answer_text = str(row["target_text"]).strip()
    eos = tokenizer.eos_token or ""
    full_text = prompt_text + answer_text + eos

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    answer_token_count = sum(label != -100 for label in labels)

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "prompt_tokens": prompt_len,
        "answer_token_count_unshifted": answer_token_count,
        "truncated": len(full_ids) >= max_length,
        "row": row,
    }


class NLLDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_length: int, use_chat_template: bool):
        self.examples = [encode_row(tokenizer, row, max_length, use_chat_template) for row in rows]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]


def collate_batch(tokenizer, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(len(example["input_ids"]) for example in examples)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    labels = []
    rows = []
    prompt_tokens = []
    truncated = []
    for example in examples:
        length = len(example["input_ids"])
        pad_len = max_len - length
        input_ids.append(example["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(example["attention_mask"] + [0] * pad_len)
        labels.append(example["labels"] + [-100] * pad_len)
        rows.append(example["row"])
        prompt_tokens.append(example["prompt_tokens"])
        truncated.append(example["truncated"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "rows": rows,
        "prompt_tokens": prompt_tokens,
        "truncated": truncated,
    }


def batch_nll(model, batch: Dict[str, Any], device: str) -> List[Dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    shift_logits = output.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = shift_labels.ne(-100)
    results: List[Dict[str, float]] = []
    for idx in range(input_ids.shape[0]):
        token_count = int(mask[idx].sum().item())
        total_nll = float(losses[idx][mask[idx]].sum().detach().cpu().item()) if token_count else 0.0
        results.append(
            {
                "answer_nll": total_nll,
                "answer_token_count": token_count,
                "answer_nll_per_token": total_nll / token_count if token_count else None,
            }
        )
    return results


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["interface_name"]].append(record)

    by_interface: Dict[str, Any] = {}
    for interface_name, items in grouped.items():
        valid = [item for item in items if item["answer_token_count"] > 0]
        total_nll = sum(item["answer_nll"] for item in valid)
        total_tokens = sum(item["answer_token_count"] for item in valid)
        token_nll = total_nll / total_tokens if total_tokens else 0.0
        by_interface[interface_name] = {
            "n": len(items),
            "valid_n": len(valid),
            "skipped_n": len(items) - len(valid),
            "mean_answer_nll": mean(item["answer_nll"] for item in valid) if valid else 0.0,
            "mean_answer_nll_per_token": mean(item["answer_nll_per_token"] for item in valid)
            if valid
            else 0.0,
            "token_weighted_nll_per_token": token_nll,
            "token_weighted_ppl": math.exp(min(token_nll, 50.0)) if total_tokens else 0.0,
            "mean_answer_tokens": mean(item["answer_token_count"] for item in valid) if valid else 0.0,
            "mean_prompt_tokens": mean(item["prompt_tokens"] for item in items) if items else 0.0,
            "truncated_count": sum(item["truncated"] for item in items),
        }

    paired = summarize_paired_utility(records)
    return {
        "metrics_by_interface": by_interface,
        "paired_context_utility_vs_no_context": paired,
    }


def summarize_paired_utility(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_source: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["answer_token_count"] <= 0:
            continue
        by_source[record["source_id"]][record["interface_name"]] = record

    utilities: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for source_id, items in by_source.items():
        baseline = items.get("no_context")
        if baseline is None:
            continue
        for interface_name, item in items.items():
            if interface_name == "no_context":
                continue
            utilities[interface_name].append(
                {
                    "source_id": source_id,
                    "utility_total_nll": baseline["answer_nll"] - item["answer_nll"],
                    "utility_nll_per_token": baseline["answer_nll_per_token"]
                    - item["answer_nll_per_token"],
                }
            )

    summary: Dict[str, Any] = {}
    for interface_name, values in utilities.items():
        summary[interface_name] = {
            "paired_n": len(values),
            "mean_utility_total_nll": mean(item["utility_total_nll"] for item in values)
            if values
            else 0.0,
            "mean_utility_nll_per_token": mean(item["utility_nll_per_token"] for item in values)
            if values
            else 0.0,
            "positive_utility_rate": mean(
                1.0 if item["utility_total_nll"] > 0 else 0.0 for item in values
            )
            if values
            else 0.0,
        }
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(summary: Dict[str, Any], output_dir: Path) -> None:
    print("HotpotQA answer NLL")
    print(f"- output_dir: {output_dir}")
    for interface_name, metrics in summary["metrics_by_interface"].items():
        print(
            f"- {interface_name}: "
            f"n={metrics['valid_n']}, "
            f"nll/token={metrics['token_weighted_nll_per_token']:.4f}, "
            f"mean_nll={metrics['mean_answer_nll']:.3f}, "
            f"prompt_tokens={metrics['mean_prompt_tokens']:.1f}, "
            f"truncated={metrics['truncated_count']}"
        )
    utilities = summary.get("paired_context_utility_vs_no_context", {})
    if utilities:
        print("- paired context utility vs no_context:")
        for interface_name, metrics in utilities.items():
            print(
                f"  - {interface_name}: "
                f"paired_n={metrics['paired_n']}, "
                f"utility_nll/token={metrics['mean_utility_nll_per_token']:.4f}, "
                f"positive_rate={metrics['positive_utility_rate']:.3f}"
            )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    run_name = args.run_name or args.model_id.replace("/", "__")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data_dir, args.split, args.interfaces, args.max_examples_per_interface)
    tokenizer, model = load_model(args.model_id, args.adapter_path, args.device, args.local_files_only)
    dataset = NLLDataset(
        rows,
        tokenizer,
        args.max_length,
        use_chat_template=not args.no_chat_template,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
    )

    records: List[Dict[str, Any]] = []
    start = time.time()
    seen = 0
    for batch in loader:
        nlls = batch_nll(model, batch, args.device)
        for row, prompt_tokens, truncated, nll in zip(
            batch["rows"],
            batch["prompt_tokens"],
            batch["truncated"],
            nlls,
        ):
            records.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "interface_name": row["interface_name"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "prompt_tokens": int(prompt_tokens),
                    "truncated": bool(truncated),
                    "metadata": row.get("metadata", {}),
                    **nll,
                }
            )
        seen += len(batch["rows"])
        if seen % args.log_every == 0 or seen == len(dataset):
            print(f"scored {seen}/{len(dataset)}")

    metrics = summarize(records)
    summary = {
        "model_id": args.model_id,
        "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
        "data_dir": str(args.data_dir),
        "split": args.split,
        "interfaces": args.interfaces,
        "max_examples_per_interface": args.max_examples_per_interface,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": args.device,
        "elapsed_seconds": time.time() - start,
        **metrics,
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "answer_nll.jsonl", records)
    print_summary(summary, output_dir)


if __name__ == "__main__":
    main()
