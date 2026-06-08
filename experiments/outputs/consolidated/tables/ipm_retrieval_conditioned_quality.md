# Retrieval-Conditioned Reader Quality

Rows split Qwen14B n=4000 top-5 predictions by whether the retrieved
window contains every annotated support unit for the same validation
example. Raw F1 is computed on the same subset, so the final column
separates coverage failures from reader behavior when support is present.

|Dataset|Support status|n|Share|Mean supp. frac.|Raw F1|Top-5 F1|Top-5 - raw|
|---|---|---|---|---|---|---|---|
|HotpotQA|Complete support|284|94.7%|1.000|0.786|0.786|+0.000|
|HotpotQA|Missing support|16|5.3%|0.500|0.671|0.448|-0.223|
|2WikiMultiHopQA|Complete support|172|57.3%|1.000|0.791|0.786|-0.006|
|2WikiMultiHopQA|Missing support|128|42.7%|0.559|0.771|0.497|-0.273|
|MuSiQue|Complete support|106|35.3%|1.000|0.621|0.709|+0.088|
|MuSiQue|Missing support|194|64.7%|0.484|0.559|0.323|-0.236|
