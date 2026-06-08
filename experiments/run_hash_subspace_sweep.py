#!/usr/bin/env python
"""Run a structured hash-projection subspace sweep for HotpotQA LoRA."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data" / "hotpotqa_interfaces" / "pilot"
DEFAULT_LORA_DIR = EXPERIMENTS / "outputs" / "lora_pilot"
DEFAULT_EVAL_DIR = EXPERIMENTS / "outputs" / "smoke_test"
DEFAULT_SUMMARY_PATH = (
    EXPERIMENTS / "outputs" / "subspace_sweep" / "hotpotqa_hash_subspace_sweep_summary.json"
)
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


INTERFACE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "raw_context": {
        "slug": "raw",
        "max_length": 3072,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "gold_supporting_sentences": {
        "slug": "gold",
        "max_length": 512,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
    },
    "gold_plus_3_distractors": {
        "slug": "gold_plus3",
        "max_length": 2048,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_support_first": {
        "slug": "raw_support_first",
        "max_length": 3072,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HotpotQA hash-projection subspace sweep.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--interfaces",
        nargs="+",
        default=["gold_supporting_sentences", "raw_context"],
    )
    parser.add_argument("--dims", nargs="+", type=int, default=[65536, 131072, 262144])
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--train-eval-size", type=int, default=100)
    parser.add_argument("--generation-eval-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(cmd: List[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def dim_slug(dim: int) -> str:
    return str(dim)


def run_name(interface_name: str, dim: int, seed: int, lr: float) -> str:
    slug = INTERFACE_CONFIGS[interface_name]["slug"]
    lr_text = f"{lr:g}"
    if lr_text.startswith("0."):
        lr_slug = "0" + lr_text[2:]
    else:
        lr_slug = lr_text.replace(".", "p").replace("-", "m")
    return f"{slug}_hash{dim_slug(dim)}_seed{seed}_lr{lr_slug}"


def eval_run_name(interface_name: str, dim: int, seed: int, lr: float) -> str:
    return f"{run_name(interface_name, dim, seed, lr)}_eval300"


def train_summary_path(args: argparse.Namespace, interface_name: str, dim: int) -> Path:
    return args.lora_dir / run_name(interface_name, dim, args.seed, args.learning_rate) / "train_summary.json"


def adapter_path(args: argparse.Namespace, interface_name: str, dim: int) -> Path:
    return args.lora_dir / run_name(interface_name, dim, args.seed, args.learning_rate) / "adapter"


def eval_summary_path(args: argparse.Namespace, interface_name: str, dim: int) -> Path:
    return args.eval_dir / eval_run_name(interface_name, dim, args.seed, args.learning_rate) / "summary.json"


def train_cmd(args: argparse.Namespace, interface_name: str, dim: int) -> List[str]:
    cfg = INTERFACE_CONFIGS[interface_name]
    return [
        sys.executable,
        str(EXPERIMENTS / "train_hotpotqa_lora.py"),
        "--data-dir",
        str(args.data_dir),
        "--interface-name",
        interface_name,
        "--model-id",
        args.model_id,
        "--train-size",
        str(args.train_size),
        "--eval-size",
        str(args.train_eval_size),
        "--max-length",
        str(cfg["max_length"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--gradient-accumulation-steps",
        str(cfg["gradient_accumulation_steps"]),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--optimizer",
        args.optimizer,
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--active-subspace-dim",
        str(dim),
        "--subspace-method",
        "hash_projection",
        "--subspace-seed",
        str(args.seed),
        "--device",
        args.device,
        "--run-name",
        run_name(interface_name, dim, args.seed, args.learning_rate),
    ]


def eval_cmd(args: argparse.Namespace, interface_name: str, dim: int) -> List[str]:
    return [
        sys.executable,
        str(EXPERIMENTS / "run_hotpotqa_smoke_test.py"),
        "--data-dir",
        str(args.data_dir),
        "--split",
        "validation",
        "--model-id",
        args.model_id,
        "--adapter-path",
        str(adapter_path(args, interface_name, dim)),
        "--interfaces",
        interface_name,
        "--max-examples-per-interface",
        str(args.generation_eval_size),
        "--device",
        args.device,
        "--run-name",
        eval_run_name(interface_name, dim, args.seed, args.learning_rate),
    ]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for interface_name in args.interfaces:
        for dim in args.dims:
            train_path = train_summary_path(args, interface_name, dim)
            eval_path = eval_summary_path(args, interface_name, dim)
            if not train_path.exists() or not eval_path.exists():
                continue
            train_summary = load_json(train_path)
            eval_summary = load_json(eval_path)
            metrics = eval_summary["metrics_by_interface"][interface_name]
            rows.append(
                {
                    "interface_name": interface_name,
                    "method": train_summary["subspace"]["method"],
                    "active_dim": dim,
                    "used_dim": train_summary["subspace"].get("used_dim"),
                    "train_size": train_summary["train_size"],
                    "updates": train_summary["updates"],
                    "optimizer": train_summary.get("optimizer"),
                    "learning_rate": train_summary["learning_rate"],
                    "initial_eval_loss": train_summary["initial_eval_loss"],
                    "final_eval_loss": train_summary["final_eval_loss"],
                    "exact_match": metrics["exact_match"],
                    "f1": metrics["f1"],
                    "answer_contained": metrics["answer_contained"],
                    "avg_prompt_tokens": metrics["avg_prompt_tokens"],
                    "truncated_count": metrics["truncated_count"],
                    "elapsed_seconds": train_summary["elapsed_seconds"],
                    "adapter_path": train_summary["adapter_path"],
                    "eval_summary_path": str(eval_path),
                }
            )
    return {
        "model_id": args.model_id,
        "method": "hash_projection",
        "seed": args.seed,
        "train_size": args.train_size,
        "train_eval_size": args.train_eval_size,
        "generation_eval_size": args.generation_eval_size,
        "dims": args.dims,
        "interfaces": args.interfaces,
        "rows": sorted(rows, key=lambda row: (row["interface_name"], row["active_dim"])),
    }


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_table(summary: Dict[str, Any]) -> None:
    print("\nHash subspace sweep summary")
    print("interface,D,loss,EM,F1,contains,time_s")
    for row in summary["rows"]:
        print(
            f"{row['interface_name']},{row['active_dim']},"
            f"{row['final_eval_loss']:.4f},"
            f"{row['exact_match']:.3f},"
            f"{row['f1']:.3f},"
            f"{row['answer_contained']:.3f},"
            f"{row['elapsed_seconds']:.1f}"
        )


def main() -> None:
    args = parse_args()
    for interface_name in args.interfaces:
        if interface_name not in INTERFACE_CONFIGS:
            raise KeyError(f"Unknown interface: {interface_name}")
        for dim in args.dims:
            if args.force_train or not train_summary_path(args, interface_name, dim).exists():
                run_command(train_cmd(args, interface_name, dim), args.dry_run)
            else:
                print(f"skip train: {train_summary_path(args, interface_name, dim)}", flush=True)

            if args.force_eval or not eval_summary_path(args, interface_name, dim).exists():
                run_command(eval_cmd(args, interface_name, dim), args.dry_run)
            else:
                print(f"skip eval: {eval_summary_path(args, interface_name, dim)}", flush=True)

    if not args.dry_run:
        summary = summarize(args)
        write_summary(args.summary_path, summary)
        print_table(summary)
        print(f"\nwrote {args.summary_path}")


if __name__ == "__main__":
    main()
