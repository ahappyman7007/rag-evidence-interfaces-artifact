# Retrieval Recall, Quality, and Cost

This table joins validation support recall with the current Qwen14B
n=4000 main result table. It is intended for the retrieval
quality/cost section.

|Dataset|Top-5 method|All support@5|Raw F1|Top-5 F1|Top-5 - raw|Raw tokens|Top-5 tokens|Token reduction|
|---|---|---|---|---|---|---|---|---|
|HotpotQA|Fine-tuned cross-encoder|0.948|0.780|0.768|-0.012|1484|685|53.8%|
|2WikiMultiHopQA|Cross-encoder|0.597|0.782|0.663|-0.120|1059|624|41.1%|
|MuSiQue|Cross-encoder|0.313|0.581|0.459|-0.121|2473|700|71.7%|
