# Frozen Qwen14B HotpotQA anchor

Non-adapted `Qwen/Qwen2.5-Coder-14B-Instruct`, greedy generation, 300 HotpotQA validation examples per interface, max input length 3072, max new tokens 32. This is a targeted reviewer-risk anchor rather than a replacement for the adapted-reader matrix.

| Dataset | Interface | EM | F1 | Answer contained | Avg prompt tokens | Trunc. |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | no context | 0.143 | 0.219 | 0.193 | 60.7 | 0 |
| HotpotQA | raw context | 0.460 | 0.630 | 0.603 | 1484.3 | 1 |
| HotpotQA | realistic top-5 | 0.477 | 0.626 | 0.617 | 685.4 | 0 |
| HotpotQA | oracle support | 0.557 | 0.738 | 0.663 | 156.2 | 0 |

Summary: raw context strongly improves over no context, the realistic top-5 window is nearly tied with raw context while using less than half the prompt tokens, and oracle support is clearly strongest. This supports the paper direction on HotpotQA without making a broader zero-shot claim across all datasets.
