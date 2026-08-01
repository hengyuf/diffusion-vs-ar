# Sudoku datasets

`raw/` contains source downloads. `processed/` contains CSV files directly
loadable by this repository through `dataset_info.json`.

Every processed CSV begins with the compatible `quizzes,solutions` columns and
preserves: `source,dataset,official_rating,rating_type,difficulty_bucket,clues,split`.

Difficulty buckets:

- Sudoku Extreme (`tdoku_backtracks`): `r0`, `r1_4`, `r5_19`, `r20_49`,
  `r50_99`, `r100_plus`.
- Kaggle 3M (`mean_search_depth_10_runs`): `r0`, `r0_1`, `r1_2`, `r2_4`,
  `r4_plus`.
- Sudoku Exchange (`sukaku_explainer`): `easy`, `medium`, `hard`,
  `diabolical`.

Ratings from different solvers are not numerically comparable. Mix bucket file
names rather than comparing their raw values.

Example curriculum mixture:

```bash
--dataset sudoku_extreme_train_r5_19,sudoku_extreme_train_r20_49,sudoku_extreme_train_r50_99 \
--mix_strategy interleave_over \
--interleave_probs 0.2,0.4,0.4
```

Rebuild processed data after downloading the sources with:

```bash
python3 tools/prepare_sudoku_datasets.py
```
