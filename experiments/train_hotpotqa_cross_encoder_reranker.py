#!/usr/bin/env python
"""Fine-tune a cross-encoder reranker on HotpotQA support-title labels."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from datasets import Dataset
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from build_hotpotqa_cross_encoder_retrieval_interfaces import paragraph_block
from build_hotpotqa_interfaces import select_examples
from inspect_hotpotqa import context_docs, load_split, supporting_facts


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "rerankers"
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune HotpotQA cross-encoder reranker.")
    parser.add_argument("--dataset-name", default="hotpotqa/hotpot_qa")
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="hotpotqa_crossenc_support_title_n5000")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def support_titles_from_facts(facts: Sequence[Tuple[str, int]]) -> List[str]:
    support_titles: List[str] = []
    seen = set()
    for title, _sent_id in facts:
        if title not in seen:
            support_titles.append(str(title))
            seen.add(title)
    return support_titles


def make_examples(ds: Dataset, size: int, seed: int, shuffle: bool) -> List[InputExample]:
    selected = select_examples(ds, size, seed, shuffle=shuffle)
    examples: List[InputExample] = []
    positives = 0
    negatives = 0
    for row in selected:
        question = str(row.get("question", "")).strip()
        docs = context_docs(row)
        support_set = set(support_titles_from_facts(supporting_facts(row)))
        for doc in docs:
            label = float(str(doc["title"]) in support_set)
            positives += int(label == 1.0)
            negatives += int(label == 0.0)
            examples.append(InputExample(texts=[question, paragraph_block(doc)], label=label))
    print(f"built {len(examples)} pairs: positives={positives}, negatives={negatives}")
    return examples


@torch.inference_mode()
def eval_pair_auc(model, tokenizer, examples: List[InputExample], batch_size: int, device: str, max_length: int) -> Dict[str, float]:
    if not examples:
        return {"pair_auc": 0.0}
    labels = torch.tensor([example.label for example in examples], dtype=torch.float32)
    scores_list: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        encoded = tokenizer(
            [example.texts[0] for example in batch],
            [example.texts[1] for example in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits.squeeze(-1).detach().cpu()
        scores_list.append(logits)
    scores = torch.cat(scores_list).float()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {"pair_auc": 0.0}
    # AUC as P(score_pos > score_neg) + 0.5 P(tie), computed exactly enough for the small dev set.
    comparisons = (pos[:, None] > neg[None, :]).float().mean()
    ties = (pos[:, None] == neg[None, :]).float().mean()
    return {"pair_auc": float(comparisons + 0.5 * ties)}


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[InputExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> InputExample:
        return self.examples[index]


def make_collate(tokenizer, max_length: int):
    def collate(batch: List[InputExample]) -> Dict[str, torch.Tensor]:
        encoded = tokenizer(
            [example.texts[0] for example in batch],
            [example.texts[1] for example in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([example.label for example in batch], dtype=torch.float32)
        return encoded

    return collate


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / args.run_name

    train = load_split(args.dataset_name, args.config, "train")
    validation = load_split(args.dataset_name, args.config, "validation")
    train_examples = make_examples(train, args.train_size, args.seed, shuffle=not args.no_shuffle)
    dev_examples = make_examples(validation, args.dev_size, args.seed, shuffle=not args.no_shuffle)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=1,
        local_files_only=args.local_files_only,
    )
    model.to(args.device)

    initial_metrics = eval_pair_auc(model, tokenizer, dev_examples, args.batch_size, args.device, args.max_length)
    train_dataloader = DataLoader(
        PairDataset(train_examples),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=make_collate(tokenizer, args.max_length),
    )

    positives = sum(1 for example in train_examples if example.label == 1.0)
    negatives = max(len(train_examples) - positives, 1)
    pos_weight = torch.tensor([negatives / max(positives, 1)], device=args.device)
    loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(len(train_dataloader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps // 2),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))

    start = time.time()
    model.train()
    step = 0
    running_loss = 0.0
    for epoch in range(args.epochs):
        for batch in train_dataloader:
            labels = batch.pop("labels").to(args.device)
            batch = {key: value.to(args.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                logits = model(**batch).logits.squeeze(-1)
                loss = loss_fct(logits.float(), labels.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running_loss += float(loss.detach().cpu())
            if step == 1 or step % 100 == 0 or step == total_steps:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={step}/{total_steps} "
                    f"loss={running_loss / step:.4f}",
                    flush=True,
                )
    elapsed = time.time() - start
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    final_metrics = eval_pair_auc(model, tokenizer, dev_examples, args.batch_size, args.device, args.max_length)

    summary = {
        "model_id": args.model_id,
        "run_name": args.run_name,
        "output_path": str(run_dir),
        "train_size": args.train_size,
        "dev_size": args.dev_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_length": args.max_length,
        "device": args.device,
        "elapsed_seconds": elapsed,
        "train_loss_mean": running_loss / max(step, 1),
        "pos_weight": float(pos_weight.item()),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
    }
    summary_path = run_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
