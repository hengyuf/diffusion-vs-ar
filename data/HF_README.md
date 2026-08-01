---
license: other
task_categories:
- question-answering
language:
- en
tags:
- sudoku
- reasoning
- planning
- discrete-diffusion
pretty_name: Diffusion vs AR Hard Sudoku
---

# Diffusion vs AR Hard Sudoku

This repository packages 8,148,696 Sudoku examples in the CSV format expected
by `HKUNLP/diffusion-vs-ar`, plus its original 100k/1k easy baseline.

Every processed file has these columns:

| column | meaning |
|---|---|
| `quizzes` | 81 row-major digits; `0` is an empty cell |
| `solutions` | complete 81-digit solution |
| `source` | original collection |
| `dataset` | normalized dataset family |
| `official_rating` | rating supplied by the source |
| `rating_type` | semantics of that rating |
| `difficulty_bucket` | categorical bucket used for mixtures |
| `clues` | number of givens |
| `split` | source-preserving train/test designation |

## Dataset families

- **Sudoku Extreme** (4,254,780): `tdoku_backtracks`; official train/test
  split is preserved. Buckets: `r0`, `r1_4`, `r5_19`, `r20_49`, `r50_99`,
  `r100_plus`.
- **Radcliffe Kaggle 3M** (3,000,000):
  `mean_search_depth_10_runs`. Buckets: `r0`, `r1_2`, `r2_4`, `r4_plus`.
- **Sudoku Exchange** (893,916): `sukaku_explainer`. Buckets: `easy`,
  `medium`, `hard`, `diabolical`. Solutions were generated with a bitmask MRV
  solver; the source states that all puzzles have a unique solution.

Raw ratings from different families are not numerically comparable.

## Licensing and provenance

- Sudoku Exchange dedicates its puzzle bank to the public domain.
- Radcliffe's 3M dataset is published as CC0.
- Sudoku Extreme combines several community benchmark sources and does not
  declare one uniform license in its data card. Review its listed upstream
  sources before redistribution outside research use.

See `processed/manifest.json` for exact per-file counts and the accompanying
GitHub repository for conversion and validation code.

