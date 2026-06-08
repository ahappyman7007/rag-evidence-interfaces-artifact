# Paired Bootstrap Uncertainty

Paired by validation `source_id`. The reported difference is
`comparison - baseline` on per-example F1.

|Dataset|Comparison|Baseline F1|Comparison F1|Diff|95% CI|P(diff > 0)|
|---|---|---:|---:|---:|---:|---:|
|HotpotQA|Support first - Raw context|0.780|0.806|+0.026|[+0.001, +0.053]|0.978|
|HotpotQA|FT cross-enc top-5 - Raw context|0.780|0.768|-0.012|[-0.045, +0.021]|0.237|
|HotpotQA|Oracle support - Raw context|0.780|0.796|+0.016|[-0.020, +0.053]|0.804|
|2WikiMultiHopQA|Support first - Raw context|0.782|0.822|+0.039|[+0.007, +0.073]|0.990|
|2WikiMultiHopQA|Cross-enc top-5 - Raw context|0.782|0.663|-0.120|[-0.170, -0.070]|0.000|
|2WikiMultiHopQA|Oracle support - Raw context|0.782|0.844|+0.061|[+0.020, +0.104]|0.999|
|2WikiMultiHopQA|Structured triples - Raw context|0.782|0.993|+0.211|[+0.168, +0.255]|1.000|
|MuSiQue|Support first - Raw context|0.581|0.651|+0.070|[+0.034, +0.106]|1.000|
|MuSiQue|Cross-enc top-5 - Raw context|0.581|0.459|-0.121|[-0.177, -0.066]|0.000|
|MuSiQue|Oracle support - Raw context|0.581|0.747|+0.166|[+0.122, +0.211]|1.000|
