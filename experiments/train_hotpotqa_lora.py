#!/usr/bin/env python
"""Train a small LoRA adapter on one exported HotpotQA interface."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "hotpotqa_interfaces" / "pilot"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "lora_pilot"
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HotpotQA LoRA pilot.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--interface-name", default="gold_supporting_sentences")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--active-subspace-dim",
        type=int,
        default=0,
        help="If >0, only this many random LoRA coordinates receive gradients.",
    )
    parser.add_argument(
        "--subspace-method",
        choices=["coordinate_mask", "hash_projection"],
        default="coordinate_mask",
        help="How to restrict the LoRA update when --active-subspace-dim > 0.",
    )
    parser.add_argument(
        "--subspace-seed",
        type=int,
        default=None,
        help="Seed for selecting active LoRA coordinates. Defaults to --seed.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def make_prompt(tokenizer, input_text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return input_text.rstrip() + "\nAnswer:"


def encode_row(tokenizer, row: Dict[str, Any], max_length: int) -> Dict[str, Any]:
    prompt_text = make_prompt(tokenizer, row["input_text"])
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
    if all(label == -100 for label in labels):
        labels[-1] = full_ids[-1]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "source_id": row.get("source_id"),
        "interface_name": row.get("interface_name"),
    }


class HotpotQALoraDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_length: int):
        self.examples = [encode_row(tokenizer, row, max_length) for row in rows]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]


def collate_batch(tokenizer, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_len = max(len(example["input_ids"]) for example in examples)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    labels = []

    for example in examples:
        length = len(example["input_ids"])
        pad_len = max_len - length
        input_ids.append(example["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(example["attention_mask"] + [0] * pad_len)
        labels.append(example["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def load_tokenizer(model_id: str, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_lora_model(args: argparse.Namespace):
    model_kwargs: Dict[str, Any] = {"local_files_only": args.local_files_only}
    if args.device.startswith("cuda"):
        model_kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.to(args.device)
    model.train()
    return model


def apply_random_coordinate_subspace(model, active_dim: int, seed: int) -> Dict[str, Any]:
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    total = sum(param.numel() for _name, param in trainable)
    if active_dim <= 0 or active_dim >= total:
        return {
            "method": "full_lora",
            "seed": seed,
            "active_dim": total,
            "total_trainable_params": total,
            "masked": False,
        }

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected = torch.randperm(total, generator=generator)[:active_dim].sort().values

    selected_cursor = 0
    offset = 0
    active_by_parameter: Dict[str, int] = {}
    for name, param in trainable:
        count = param.numel()
        end = offset + count
        start_cursor = selected_cursor
        while selected_cursor < active_dim and int(selected[selected_cursor]) < end:
            selected_cursor += 1

        local_count = selected_cursor - start_cursor
        active_by_parameter[name] = local_count
        if local_count > 0:
            local_indices = selected[start_cursor:selected_cursor] - offset
            flat_mask = torch.zeros(count, dtype=torch.bool)
            flat_mask[local_indices] = True
            mask = flat_mask.view_as(param).to(param.device)
        else:
            mask = torch.zeros_like(param, dtype=torch.bool, device=param.device)

        def make_hook(active_mask):
            return lambda grad: grad * active_mask.to(dtype=grad.dtype)

        param.register_hook(make_hook(mask))
        offset = end

    return {
        "method": "random_coordinate_subspace",
        "seed": seed,
        "active_dim": active_dim,
        "total_trainable_params": total,
        "active_fraction": active_dim / total,
        "masked": True,
        "active_by_parameter": active_by_parameter,
    }


class HashGradientProjector:
    """Project full LoRA gradients into a signed hash random subspace."""

    def __init__(self, model, active_dim: int, seed: int):
        self.trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        self.active_dim = active_dim
        self.device = self.trainable[0][1].device
        self.state: Dict[str, Dict[str, torch.Tensor]] = {}

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        counts = torch.zeros(active_dim, dtype=torch.float32)
        active_by_parameter: Dict[str, int] = {}
        total = 0

        for name, param in self.trainable:
            count = param.numel()
            total += count
            buckets = torch.randint(active_dim, (count,), generator=generator, dtype=torch.long)
            signs = torch.randint(0, 2, (count,), generator=generator, dtype=torch.int8)
            signs = signs.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
            counts.scatter_add_(0, buckets, torch.ones(count, dtype=torch.float32))
            self.state[name] = {
                "buckets_cpu": buckets,
                "signs_cpu": signs,
            }
            active_by_parameter[name] = int(torch.unique(buckets).numel())

        self.inv_sqrt_counts = counts.clamp_min(1.0).rsqrt().to(self.device)
        for item in self.state.values():
            item["buckets"] = item.pop("buckets_cpu").to(self.device)
            item["signs"] = item.pop("signs_cpu").to(self.device)

        used_dims = int((counts > 0).sum().item())
        self.info = {
            "method": "hash_projection",
            "seed": seed,
            "active_dim": active_dim,
            "used_dim": used_dims,
            "total_trainable_params": total,
            "active_fraction": active_dim / total,
            "masked": True,
            "active_by_parameter": active_by_parameter,
        }

    @torch.no_grad()
    def project_gradients(self) -> None:
        accum = torch.zeros(self.active_dim, dtype=torch.float32, device=self.device)
        for name, param in self.trainable:
            if param.grad is None:
                continue
            item = self.state[name]
            buckets = item["buckets"]
            signs = item["signs"]
            inv = self.inv_sqrt_counts.index_select(0, buckets)
            grad = param.grad.detach().flatten().float()
            accum.scatter_add_(0, buckets, grad * signs * inv)

        for name, param in self.trainable:
            if param.grad is None:
                continue
            item = self.state[name]
            buckets = item["buckets"]
            signs = item["signs"]
            inv = self.inv_sqrt_counts.index_select(0, buckets)
            projected = signs * inv * accum.index_select(0, buckets)
            param.grad.detach().copy_(projected.view_as(param).to(dtype=param.grad.dtype))


def configure_subspace(model, active_dim: int, seed: int, method: str):
    trainable_total = sum(param.numel() for _name, param in model.named_parameters() if param.requires_grad)
    if active_dim <= 0 or active_dim >= trainable_total:
        return {
            "method": "full_lora",
            "seed": seed,
            "active_dim": trainable_total,
            "total_trainable_params": trainable_total,
            "masked": False,
        }, None

    if method == "coordinate_mask":
        return apply_random_coordinate_subspace(model, active_dim, seed), None
    if method == "hash_projection":
        projector = HashGradientProjector(model, active_dim, seed)
        return projector.info, projector
    raise ValueError(f"unknown subspace method: {method}")


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: str) -> float:
    model.eval()
    losses: List[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))
    model.train()
    return float(mean(losses)) if losses else 0.0


def train(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    run_name = args.run_name or f"{args.interface_name}_r{args.lora_r}_n{args.train_size}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / f"train__{args.interface_name}.jsonl"
    eval_path = args.data_dir / f"validation__{args.interface_name}.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    tokenizer = load_tokenizer(args.model_id, args.local_files_only)
    train_rows = read_jsonl(train_path, args.train_size)
    eval_rows = read_jsonl(eval_path, args.eval_size)
    train_dataset = HotpotQALoraDataset(train_rows, tokenizer, args.max_length)
    eval_dataset = HotpotQALoraDataset(eval_rows, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
    )

    model = load_lora_model(args)
    model.print_trainable_parameters()
    subspace_seed = args.subspace_seed if args.subspace_seed is not None else args.seed
    subspace_info, gradient_projector = configure_subspace(
        model,
        active_dim=args.active_subspace_dim,
        seed=subspace_seed,
        method=args.subspace_method,
    )
    if subspace_info["masked"]:
        print(
            f"active {subspace_info['method']} subspace: "
            f"{subspace_info['active_dim']}/"
            f"{subspace_info['total_trainable_params']} "
            f"({subspace_info['active_fraction']:.6f})"
        )

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    initial_eval_loss = evaluate_loss(model, eval_loader, args.device)
    print(f"initial_eval_loss={initial_eval_loss:.4f}")

    optimizer.zero_grad(set_to_none=True)
    update_step = 0
    running_losses: List[float] = []
    start_time = time.time()

    target_micro_steps = math.ceil(len(train_loader) * args.epochs)
    micro_step = 0
    while micro_step < target_micro_steps:
        for batch in train_loader:
            micro_step += 1
            if micro_step > target_micro_steps:
                break
            batch = move_batch(batch, args.device)
            output = model(**batch)
            loss = output.loss / args.gradient_accumulation_steps
            loss.backward()
            running_losses.append(float(output.loss.detach().cpu()))

            do_update = (
                micro_step % args.gradient_accumulation_steps == 0
                or micro_step == target_micro_steps
            )
            if not do_update:
                continue

            if gradient_projector is not None:
                gradient_projector.project_gradients()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1

            if update_step % args.log_every == 0 or update_step == 1 or update_step == total_updates:
                recent = running_losses[-args.log_every * args.gradient_accumulation_steps :]
                lr = scheduler.get_last_lr()[0]
                print(
                    f"update={update_step}/{total_updates} "
                    f"loss={mean(recent):.4f} lr={lr:.2e}"
                )

    final_eval_loss = evaluate_loss(model, eval_loader, args.device)
    print(f"final_eval_loss={final_eval_loss:.4f}")

    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")

    summary = {
        "model_id": args.model_id,
        "interface_name": args.interface_name,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.target_modules,
        },
        "subspace": subspace_info,
        "updates": update_step,
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "elapsed_seconds": time.time() - start_time,
        "adapter_path": str(output_dir / "adapter"),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("HotpotQA LoRA training summary")
    print(f"- interface: {summary['interface_name']}")
    print(f"- train_size: {summary['train_size']}")
    print(f"- updates: {summary['updates']}")
    print(f"- initial_eval_loss: {summary['initial_eval_loss']:.4f}")
    print(f"- final_eval_loss: {summary['final_eval_loss']:.4f}")
    print(f"- adapter_path: {summary['adapter_path']}")


def main() -> None:
    args = parse_args()
    summary = train(args)
    print_summary(summary)


if __name__ == "__main__":
    main()
