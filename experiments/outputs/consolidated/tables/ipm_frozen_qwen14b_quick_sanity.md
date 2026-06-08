# Frozen Qwen14B quick sanity check

Non-adapted `Qwen/Qwen2.5-Coder-14B-Instruct`, greedy generation, 50
validation examples per interface, max input length 3072, max new tokens 32.
This is a reviewer-risk diagnostic, not a replacement for the adapted-reader
main results.

| Dataset | Interface | EM | F1 | Answer contained | Avg prompt tokens | Trunc. |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | no context | 0.160 | 0.206 | 0.180 | 60.6 | 0 |
| HotpotQA | raw context | 0.420 | 0.596 | 0.640 | 1420.2 | 0 |
| HotpotQA | support-first | 0.480 | 0.648 | 0.680 | 1420.2 | 0 |
| HotpotQA | gold support | 0.440 | 0.603 | 0.600 | 156.8 | 0 |
| HotpotQA | realistic top-5 | 0.480 | 0.605 | 0.640 | 651.9 | 0 |
| 2WikiMultiHopQA | no context | 0.200 | 0.231 | 0.260 | 57.6 | 0 |
| 2WikiMultiHopQA | raw context | 0.480 | 0.603 | 0.640 | 1021.6 | 0 |
| 2WikiMultiHopQA | support-first | 0.520 | 0.631 | 0.660 | 1021.6 | 0 |
| 2WikiMultiHopQA | gold support | 0.560 | 0.687 | 0.720 | 145.6 | 0 |
| 2WikiMultiHopQA | realistic top-5 | 0.440 | 0.571 | 0.640 | 629.6 | 0 |
| MuSiQue | no context | 0.060 | 0.172 | 0.100 | 60.8 | 0 |
| MuSiQue | raw context | 0.240 | 0.408 | 0.340 | 2476.9 | 7 |
| MuSiQue | support-first | 0.320 | 0.440 | 0.420 | 2476.9 | 7 |
| MuSiQue | gold support | 0.380 | 0.593 | 0.580 | 406.3 | 0 |
| MuSiQue | realistic top-5 | 0.180 | 0.320 | 0.280 | 707.0 | 0 |

Summary: the frozen reader shows the same qualitative ordering as the adapted
readers in this small diagnostic. Raw context strongly beats no-context;
support-first is above raw context on all three datasets; gold support is
strongest on 2WikiMultiHopQA and MuSiQue; realistic top-5 is competitive on
HotpotQA but lower on 2WikiMultiHopQA and MuSiQue. The sample size is small, so
use this only as a sanity check.
