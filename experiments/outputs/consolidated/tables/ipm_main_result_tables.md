# IPM Consolidated Main Results

Generated from `experiments/outputs/rank_sweep/*summary.json` by `experiments/consolidate_ipm_results.py`.

## Qwen14B n=4000 Main Interface Hierarchy

|Dataset|Raw context|Support first|Realistic top-5|Oracle support|Structured triples|
|---|---:|---:|---:|---:|---:|
|HotpotQA|0.780|0.806|0.768|0.796|--|
|2WikiMultiHopQA|0.782|0.822|0.663|0.844|0.993|
|MuSiQue|0.581|0.651|0.459|0.747|--|

## Support-First Gain over Raw Context

|Dataset|Model|Train size|Raw F1|Support-first F1|Gain|
|---|---|---:|---:|---:|---:|
|HotpotQA|Llama-3.2-3B|4000|0.719|0.733|+0.014|
|HotpotQA|Qwen2.5-Coder-7B|1000|0.696|0.748|+0.052|
|HotpotQA|Mistral-7B|1000|0.684|0.765|+0.081|
|HotpotQA|Llama-3.1-8B|1000|0.721|0.760|+0.039|
|HotpotQA|Gemma-2-9B|1000|0.713|0.776|+0.062|
|HotpotQA|Qwen2.5-Coder-14B|4000|0.780|0.806|+0.026|
|2WikiMultiHopQA|Llama-3.2-1B|4000|0.475|0.523|+0.048|
|2WikiMultiHopQA|Llama-3.2-3B|4000|0.657|0.732|+0.075|
|2WikiMultiHopQA|Qwen2.5-Coder-7B|1000|0.618|0.650|+0.032|
|2WikiMultiHopQA|Mistral-7B|1000|0.585|0.700|+0.115|
|2WikiMultiHopQA|Llama-3.1-8B|1000|0.582|0.697|+0.115|
|2WikiMultiHopQA|Gemma-2-9B|1000|0.665|0.765|+0.100|
|2WikiMultiHopQA|Qwen2.5-Coder-14B|4000|0.782|0.822|+0.039|
|MuSiQue|Llama-3.2-1B|4000|0.299|0.459|+0.159|
|MuSiQue|Llama-3.2-3B|4000|0.507|0.587|+0.080|
|MuSiQue|Qwen2.5-Coder-7B|1000|0.434|0.560|+0.126|
|MuSiQue|Mistral-7B|1000|0.404|0.506|+0.102|
|MuSiQue|Llama-3.1-8B|1000|0.505|0.619|+0.115|
|MuSiQue|Qwen2.5-Coder-14B|4000|0.581|0.651|+0.070|
