#!/usr/bin/env python
"""Run and summarize a small HotpotQA LoRA rank sweep."""

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
DEFAULT_SUMMARY_PATH = EXPERIMENTS / "outputs" / "rank_sweep" / "hotpotqa_lora_rank_sweep_summary.json"
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


INTERFACE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "no_context": {
        "slug": "no_context",
        "max_length": 256,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
    },
    "raw_context": {
        "slug": "raw_context",
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
    "gold_supporting_paragraphs": {
        "slug": "gold_paragraphs",
        "max_length": 1024,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
    },
    "gold_evidence_triples": {
        "slug": "gold_triples",
        "max_length": 256,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
    },
    "gold_plus_1_distractor": {
        "slug": "gold_plus1",
        "max_length": 1024,
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
    "raw_support_first_2docs": {
        "slug": "raw_support_first_2docs",
        "max_length": 1024,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
    },
    "raw_support_first_3docs": {
        "slug": "raw_support_first_3docs",
        "max_length": 1536,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_support_first_5docs": {
        "slug": "raw_support_first_5docs",
        "max_length": 2048,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_bm25_order": {
        "slug": "raw_bm25_order",
        "max_length": 3072,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_bm25_top3docs": {
        "slug": "raw_bm25_top3docs",
        "max_length": 1536,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_bm25_top5docs": {
        "slug": "raw_bm25_top5docs",
        "max_length": 2048,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_embed_order": {
        "slug": "raw_embed_order",
        "max_length": 3072,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_embed_top3docs": {
        "slug": "raw_embed_top3docs",
        "max_length": 1536,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_embed_top5docs": {
        "slug": "raw_embed_top5docs",
        "max_length": 2048,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_crossenc_order": {
        "slug": "raw_crossenc_order",
        "max_length": 3072,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_crossenc_top3docs": {
        "slug": "raw_crossenc_top3docs",
        "max_length": 1536,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_crossenc_top5docs": {
        "slug": "raw_crossenc_top5docs",
        "max_length": 2048,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_suptitle_order": {
        "slug": "raw_suptitle_order",
        "max_length": 3072,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_suptitle_top3docs": {
        "slug": "raw_suptitle_top3docs",
        "max_length": 1536,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_suptitle_top5docs": {
        "slug": "raw_suptitle_top5docs",
        "max_length": 2048,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_ftcrossenc_order": {
        "slug": "raw_ftcrossenc_order",
        "max_length": 3072,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_ftcrossenc_top3docs": {
        "slug": "raw_ftcrossenc_top3docs",
        "max_length": 1536,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_ftcrossenc_top5docs": {
        "slug": "raw_ftcrossenc_top5docs",
        "max_length": 2048,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_support_middle": {
        "slug": "raw_support_middle",
        "max_length": 3072,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_support_shuffled": {
        "slug": "raw_support_shuffled",
        "max_length": 3072,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
    "raw_support_last": {
        "slug": "raw_support_last",
        "max_length": 3072,
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HotpotQA LoRA rank sweep.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--interfaces", nargs="+", default=list(INTERFACE_CONFIGS.keys()))
    parser.add_argument("--ranks", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--train-eval-size", type=int, default=100)
    parser.add_argument("--generation-eval-size", type=int, default=300)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora-dir", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(cmd: List[str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_name(interface_name: str, rank: int, train_size: int) -> str:
    slug = INTERFACE_CONFIGS[interface_name]["slug"]
    return f"{slug}_r{rank}_n{train_size}"


def eval_run_name(interface_name: str, rank: int, train_size: int) -> str:
    slug = INTERFACE_CONFIGS[interface_name]["slug"]
    return f"{slug}_lora_r{rank}_n{train_size}_eval300"


def train_summary_path(args: argparse.Namespace, interface_name: str, rank: int) -> Path:
    return args.lora_dir / run_name(interface_name, rank, args.train_size) / "train_summary.json"


def adapter_path(args: argparse.Namespace, interface_name: str, rank: int) -> Path:
    return args.lora_dir / run_name(interface_name, rank, args.train_size) / "adapter"


def eval_summary_path(args: argparse.Namespace, interface_name: str, rank: int) -> Path:
    return args.eval_dir / eval_run_name(interface_name, rank, args.train_size) / "summary.json"


def train_cmd(args: argparse.Namespace, interface_name: str, rank: int) -> List[str]:
    cfg = INTERFACE_CONFIGS[interface_name]
    alpha = rank * 2
    cmd = [
        sys.executable,
        str(EXPERIMENTS / "train_hotpotqa_lora.py"),
        "--data-dir",
        str(args.data_dir),
        "--interface-name",
        interface_name,
        "--model-id",
        args.model_id,
        "--output-dir",
        str(args.lora_dir),
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
        "--lora-r",
        str(rank),
        "--lora-alpha",
        str(alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--device",
        args.device,
        "--run-name",
        run_name(interface_name, rank, args.train_size),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    return cmd


def eval_cmd(args: argparse.Namespace, interface_name: str, rank: int) -> List[str]:
    cmd = [
        sys.executable,
        str(EXPERIMENTS / "run_hotpotqa_smoke_test.py"),
        "--data-dir",
        str(args.data_dir),
        "--split",
        "validation",
        "--model-id",
        args.model_id,
        "--adapter-path",
        str(adapter_path(args, interface_name, rank)),
        "--output-dir",
        str(args.eval_dir),
        "--interfaces",
        interface_name,
        "--max-examples-per-interface",
        str(args.generation_eval_size),
        "--device",
        args.device,
        "--run-name",
        eval_run_name(interface_name, rank, args.train_size),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    return cmd


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for interface_name in args.interfaces:
        for rank in args.ranks:
            train_path = train_summary_path(args, interface_name, rank)
            eval_path = eval_summary_path(args, interface_name, rank)
            if not train_path.exists() or not eval_path.exists():
                continue
            train_summary = load_json(train_path)
            eval_summary = load_json(eval_path)
            metrics = eval_summary["metrics_by_interface"][interface_name]
            rows.append(
                {
                    "interface_name": interface_name,
                    "rank": rank,
                    "alpha": train_summary["lora"]["alpha"],
                    "train_size": train_summary["train_size"],
                    "updates": train_summary["updates"],
                    "initial_eval_loss": train_summary["initial_eval_loss"],
                    "final_eval_loss": train_summary["final_eval_loss"],
                    "elapsed_seconds": train_summary["elapsed_seconds"],
                    "exact_match": metrics["exact_match"],
                    "f1": metrics["f1"],
                    "answer_contained": metrics["answer_contained"],
                    "avg_prompt_tokens": metrics["avg_prompt_tokens"],
                    "truncated_count": metrics["truncated_count"],
                    "adapter_path": train_summary["adapter_path"],
                    "eval_summary_path": str(eval_path),
                }
            )

    summary = {
        "model_id": args.model_id,
        "train_size": args.train_size,
        "train_eval_size": args.train_eval_size,
        "generation_eval_size": args.generation_eval_size,
        "ranks": args.ranks,
        "interfaces": args.interfaces,
        "rows": sorted(rows, key=lambda row: (row["interface_name"], row["rank"])),
    }
    return summary


def write_summary(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_table(summary: Dict[str, Any]) -> None:
    print("\nRank sweep summary")
    print("interface,rank,loss,EM,F1,contains,time_s")
    for row in summary["rows"]:
        print(
            f"{row['interface_name']},{row['rank']},"
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
        for rank in args.ranks:
            if args.force_train or not train_summary_path(args, interface_name, rank).exists():
                run_command(train_cmd(args, interface_name, rank), args.dry_run)
            else:
                print(f"skip train: {train_summary_path(args, interface_name, rank)}", flush=True)

            if args.force_eval or not eval_summary_path(args, interface_name, rank).exists():
                run_command(eval_cmd(args, interface_name, rank), args.dry_run)
            else:
                print(f"skip eval: {eval_summary_path(args, interface_name, rank)}", flush=True)

    if not args.dry_run:
        summary = summarize(args)
        write_summary(args.summary_path, summary)
        print_table(summary)
        print(f"\nwrote {args.summary_path}")


if __name__ == "__main__":
    main()
