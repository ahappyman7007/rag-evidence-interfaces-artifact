# Retrieval Support Recall

Computed from validation interface metadata. `All support` is the main
multi-hop retrieval diagnostic: the fraction of examples where every
support document appears within the top-k window.

|Dataset|Method|k|Any support|All support|Mean support frac.|Mean latest support rank|
|---|---|---|---|---|---|---|
|HotpotQA|BM25|5|0.994|0.702|0.848|4.54|
|HotpotQA|BM25|10|1.000|1.000|1.000|4.54|
|HotpotQA|BM25|20|1.000|1.000|1.000|4.54|
|HotpotQA|Cross-encoder|5|0.995|0.764|0.879|3.94|
|HotpotQA|Cross-encoder|10|1.000|1.000|1.000|3.94|
|HotpotQA|Cross-encoder|20|1.000|1.000|1.000|3.94|
|HotpotQA|Fine-tuned cross-encoder|5|0.999|0.948|0.974|2.63|
|HotpotQA|Fine-tuned cross-encoder|10|1.000|1.000|1.000|2.63|
|HotpotQA|Fine-tuned cross-encoder|20|1.000|1.000|1.000|2.63|
|2WikiMultiHopQA|BM25|5|0.999|0.534|0.789|5.56|
|2WikiMultiHopQA|BM25|10|1.000|1.000|1.000|5.56|
|2WikiMultiHopQA|BM25|20|1.000|1.000|1.000|5.56|
|2WikiMultiHopQA|Cross-encoder|5|1.000|0.597|0.824|4.88|
|2WikiMultiHopQA|Cross-encoder|10|1.000|1.000|1.000|4.88|
|2WikiMultiHopQA|Cross-encoder|20|1.000|1.000|1.000|4.88|
|MuSiQue|BM25|5|0.931|0.206|0.571|12.53|
|MuSiQue|BM25|10|0.975|0.412|0.717|12.53|
|MuSiQue|BM25|20|1.000|1.000|1.000|12.53|
|MuSiQue|Cross-encoder|5|0.966|0.313|0.651|10.54|
|MuSiQue|Cross-encoder|10|0.993|0.527|0.786|10.54|
|MuSiQue|Cross-encoder|20|1.000|1.000|1.000|10.54|
