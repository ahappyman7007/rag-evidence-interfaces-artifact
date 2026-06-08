# Anonymous Review Artifact

This artifact supports the anonymous IPM submission, "Evidence Interfaces for
RAG Readers." It is a curated review artifact, not a full copy of the local
research workspace.

## Reproduction Scope

The artifact supports three different levels of reproduction.

1. **Quick manuscript check.** Compile the anonymous manuscript from the
   included LaTeX source and figures.
2. **Aggregate-result reproduction.** Rebuild the paper-facing tables and
   code-generated figures from sanitized rank-sweep summaries included in this
   artifact. This is the main review-stage path and does not require GPUs.
3. **Full rerun from public resources.** Reconstruct interface files from the
   original public datasets and rerun reader adaptation/evaluation. This
   requires downloading third-party datasets and model checkpoints from their
   original providers. The artifact intentionally does not redistribute those
   files.

## What Is Included

- paper source files needed to compile the anonymous manuscript;
- paper figures and consolidated aggregate result tables;
- sanitized rank-sweep summary JSON files used to rebuild the main tables and
  figures;
- interface-construction, retrieval/reranking, reader-adaptation, evaluation,
  analysis, and plotting scripts;
- environment specification.

## What Is Not Included

Raw third-party datasets, trained adapter weights, cached model files,
prediction JSONL files with full examples, GPU scheduler logs, process IDs,
external-review prompts/outputs, and citation-checker raw reports are not part
of this artifact.

## Environment

For artifact review, use the compact environment:

```bash
conda env create -f artifact_environment.yml
conda activate rag_eid_artifact
```

The larger `rag_eid_environment.yml` records the full local research
environment used during development. The compact artifact environment is the
recommended starting point for rebuilding tables/figures, reconstructing
interfaces, and running the CPU smoke test below.

## Quick Check: Rebuild Tables And Figures

From the artifact root:

```bash
python experiments/consolidate_ipm_results.py --skip-paper-copy
```

This rebuilds the consolidated IPM tables and the three paper-facing figures
from the sanitized rank-sweep summary files under
`experiments/outputs/rank_sweep/`.

Expected outputs are written under:

```text
experiments/outputs/consolidated/tables/
experiments/outputs/consolidated/figures/
```

The main paper tables are summarized in
`experiments/outputs/consolidated/tables/ipm_main_result_tables.md`; the
appendix and diagnostic tables are available as CSV/MD files in the same
directory.

## Quick Check: Compile The Manuscript

From the artifact root:

```bash
cd paper_ipm
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

This should produce `paper_ipm/main.pdf`. During anonymous review, the PDF
metadata should have an empty `Author` field.

## Reconstructing Interface Files

The paper studies several reader-facing evidence interfaces. The following
commands show the canonical reconstruction pattern. They download the public
datasets through their official providers when possible and write derived
interface files locally. Do not redistribute derived files containing source
passages unless the source dataset license permits it.

HotpotQA:

```bash
python experiments/build_hotpotqa_interfaces.py \
  --train-size 4000 \
  --validation-size 500 \
  --seed 13 \
  --output-root data/hotpotqa_interfaces \
  --run-name hotpotqa_main \
  --separate-interface-files
```

2WikiMultiHopQA:

```bash
python experiments/build_2wiki_interfaces.py \
  --train-size 4000 \
  --validation-size 500 \
  --seed 13 \
  --output-root data/2wiki_interfaces \
  --run-name 2wiki_main \
  --separate-interface-files
```

MuSiQue:

```bash
python experiments/build_musique_interfaces.py \
  --train-size 4000 \
  --validation-size 500 \
  --seed 13 \
  --output-root data/musique_interfaces \
  --run-name musique_main \
  --separate-interface-files
```

The retrieval/reranking interface scripts follow the same pattern and are
included under `experiments/`:

- `build_hotpotqa_cross_encoder_retrieval_interfaces.py`
- `build_hotpotqa_embedding_retrieval_interfaces.py`
- `build_hotpotqa_supervised_title_reranker_interfaces.py`
- `build_2wiki_cross_encoder_retrieval_interfaces.py`
- `build_musique_cross_encoder_retrieval_interfaces.py`

## Rerunning Reader Adaptation

Reader adaptation uses LoRA rank-8 unless a specific ablation changes the
rank. The driver below shows the HotpotQA pattern; use the generated interface
directory, select the interface names to run, and choose a model identifier
available from the model provider.

```bash
python experiments/run_lora_rank_sweep.py \
  --data-dir data/hotpotqa_interfaces/hotpotqa_main \
  --model-id <MODEL_ID> \
  --interfaces raw_context raw_support_first gold_supporting_sentences \
  --ranks 8 \
  --train-size 4000 \
  --train-eval-size 300 \
  --generation-eval-size 300 \
  --device cuda \
  --lora-dir experiments/outputs/lora \
  --eval-dir experiments/outputs/eval \
  --summary-path experiments/outputs/rank_sweep/hotpotqa_r8_summary.json
```

Use `--dry-run` to inspect the resolved commands without launching training:

```bash
python experiments/run_lora_rank_sweep.py --dry-run
```

Full reruns are GPU-dependent and may take many hours. They also require the
reviewer to obtain the same public model checkpoints and datasets. The
included sanitized rank-sweep summaries are provided so reviewers can verify
the paper's aggregate analyses without rerunning all training jobs.

## CPU Smoke Test

The following commands test the main code path without GPUs or large models.
They use two HotpotQA examples and a tiny public test model. The goal is only
to verify that interface construction, LoRA training, and evaluation execute
end to end; the numbers are not paper results.

```bash
python experiments/build_hotpotqa_interfaces.py \
  --train-size 2 \
  --validation-size 2 \
  --seed 13 \
  --output-root generated/hotpotqa_interfaces \
  --run-name smoke \
  --separate-interface-files

python experiments/train_hotpotqa_lora.py \
  --data-dir generated/hotpotqa_interfaces/smoke \
  --interface-name gold_supporting_sentences \
  --model-id hf-internal-testing/tiny-random-LlamaForCausalLM \
  --output-dir generated/tiny_lora \
  --run-name smoke_tiny \
  --train-size 2 \
  --eval-size 1 \
  --max-length 128 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --epochs 1 \
  --learning-rate 1e-4 \
  --lora-r 2 \
  --lora-alpha 4 \
  --device cpu

python experiments/run_hotpotqa_smoke_test.py \
  --data-dir generated/hotpotqa_interfaces/smoke \
  --split validation \
  --model-id hf-internal-testing/tiny-random-LlamaForCausalLM \
  --adapter-path generated/tiny_lora/smoke_tiny/adapter \
  --output-dir generated/tiny_eval \
  --interfaces gold_supporting_sentences \
  --max-examples-per-interface 1 \
  --device cpu \
  --max-new-tokens 8 \
  --run-name smoke_tiny_eval
```

## Included Main Experimental Families

The sanitized rank-sweep summaries cover the experimental families reported in
the paper:

- interface hierarchy and support-first comparisons on HotpotQA,
  2WikiMultiHopQA, and MuSiQue;
- Qwen14B scaling summaries up to 4000 training examples;
- smaller-reader and cross-model sanity checks;
- retrieval-window, cross-encoder, and support-recall diagnostics;
- no-support reliance diagnostics and error taxonomy summaries;
- rank/subspace and position/distractor controls.

## Sanity And Privacy Checks

Before packaging, the artifact builder scans the curated output for local
absolute paths, private usernames, credential-like relay patterns, GPU infrastructure
strings, and build/log/checkpoint files. The packaged artifact is intended for
anonymous review and should remain separate from the non-anonymous title page
and administrative declarations.
