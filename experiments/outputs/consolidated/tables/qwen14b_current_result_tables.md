# Qwen14B Current Result Tables

Generated from `experiments/outputs/rank_sweep/*qwen14b*r8*n*_summary.json`.

## Main Scaling F1

### hotpotqa
| Interface | n=500 | n=1000 | n=2000 | n=4000 |
|---|---:|---:|---:|---:|
| `raw_context` | 0.727 | 0.730 | 0.748 | 0.780 |
| `raw_support_first` | 0.766 | 0.785 | 0.801 | 0.806 |
| `raw_ftcrossenc_top5docs` | 0.742 | 0.746 | 0.755 | 0.768 |
| `gold_supporting_sentences` | 0.761 | 0.781 | 0.787 | 0.796 |

### 2wiki
| Interface | n=1000 | n=2000 | n=4000 |
|---|---:|---:|---:|
| `raw_context` | 0.691 | 0.694 | 0.782 |
| `raw_support_first` | 0.767 | 0.772 | 0.822 |
| `raw_crossenc_top5docs` | 0.605 | 0.646 | 0.663 |
| `gold_supporting_sentences` | 0.789 | 0.816 | 0.844 |
| `gold_evidence_triples` | 0.993 | 0.993 | 0.993 |

### musique
| Interface | n=1000 | n=2000 | n=4000 |
|---|---:|---:|---:|
| `raw_context` | 0.512 | 0.562 | 0.581 |
| `raw_support_first` | 0.577 | 0.617 | 0.651 |
| `raw_crossenc_top5docs` | 0.422 | 0.437 | 0.459 |
| `gold_supporting_paragraphs` | 0.689 | 0.692 | 0.747 |

## No-Context Gap at n=2000
| Dataset | No context | Raw context | Support-first | Best gold/structured |
|---|---:|---:|---:|---:|
| hotpotqa | 0.280 | 0.748 | 0.801 | 0.787 |
| 2wiki | 0.315 | 0.694 | 0.772 | 0.993 |
| musique | 0.144 | 0.562 | 0.617 | 0.692 |

## Current Interpretation
- More data helps raw context, but support-first remains consistently better.
- No-context baselines are far below evidence-bearing interfaces, especially on MuSiQue.
- Existing 2Wiki/MuSiQue controls suggest ordering full context is often better than top-k truncation; HotpotQA controls are still running.
