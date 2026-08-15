# Decoding-path evaluation and viewer

Evaluates an MDM checkpoint on every named eval split of its training config,
records the full denoising path of each sample — including the 9-digit
probability distribution of all 81 cells at every step — and renders it as an
interactive page.

## 1. Dump the paths

```bash
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/diffusion-vs-ar

CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u tools/dump_decoding_paths.py \
  --config configs/sudoku/mdm-extreme-mix10k-train-multidifficulty-eval.yaml \
  --checkpoint output/sudoku/mdm-extreme-mix10k-train-multidifficulty-eval-tiny/checkpoint-45500 \
  --output_dir output/decoding_paths/checkpoint-45500
```

Runs in ~4 minutes on one H100 and writes ~35 MB (one 5.1 MB JSON per split,
`index.json`, and a copy of the viewer).

Useful flags: `--only extreme_r5_19,extreme_r50_99` to restrict splits (split
*indices* are preserved, so the sample draw does not change), `--num_samples` to
override `eval_num_samples`, `--seed` to change the decoder's Gumbel stream,
`--dtype float16`.

The script mirrors `mdm/workflow.py`'s eval-set construction exactly — each split
is shuffled with `eval_sample_seed + eval_index` and truncated to
`eval_num_samples` — so it evaluates **the same 200 puzzles per split that the
Trainer was scoring during training**. The generation loop is a copy of
`CustomDiffusionTrainer.generate_samples` with recording hooks; it must stay
numerically identical to it.

A `checkpoint-N/` directory saves no `config.json`, so the model config and
tokenizer are read from the config's `model_name_or_path` (override with
`--config_dir`) while the weights come from `--checkpoint`. Unlike
`core/loader.py`, which loads with `strict=False` and silently evaluates a random
model if the load fails, this script raises if the transformer weights do not land.

## 2. View

`fetch()` does not work on `file://` URLs, so serve the directory:

```bash
cd output/decoding_paths/checkpoint-45500
python3 -m http.server 8000
```

From your laptop: `ssh -L 8000:localhost:8000 pool0-01794`, then open
<http://localhost:8000/>.

**Left panel** — the puzzle and its ground-truth solution: given clues in bold
ink, the 58-or-so cells the model must fill in blue.

**Right panel** — the decoding path, animated over the 20 denoising steps.
Two cell displays:

- *9-digit probabilities* (default): each cell is a 3×3 pencil-mark heatmap of
  p(1…9) at that step on a blue ramp, the ground-truth digit outlined in green,
  and the currently committed digit pinned on top (red if it disagrees with the
  solution).
- *Current digit + confidence*: the plain board, with p(argmax) as a thin bar
  along each cell floor.

Orange rings mark cells unmasked at this step, violet rings cells **re-masked**
at this step (see below). Click or tab to any cell to get its exact
distribution, its evolution across all 20 steps as a step×digit heatmap, and a
table view with the numbers.

Filters (split, correct/incorrect, sample) sit in one row and scope everything.
Space plays/pauses; arrow keys step.

## What the path actually shows

`topk_decoding` re-scores **every** target position at every step, not just the
masked ones, and remasks the lowest-confidence `t/T` fraction. Committed cells
can therefore be thrown back to `[MASK]`, and they routinely are:

```
tools/check_decoding_dump.py output/decoding_paths/checkpoint-45500 extreme_r5_19

newly-committed cell-events : 61303
re-masked cell-events       : 45103
samples with >=1 re-mask    : 200/200      # peak ~22 committed / ~18 re-masked per step
```

So the net masked count falls by only ~3 per step while ~40 cells change state.
This churn is the interesting part of the path, which is why re-masking has its
own encoding rather than being left implicit.

## Accuracy check

`tools/check_decoding_dump.py <dump_dir> [split]` prints the per-split accuracy
against the numbers the Trainer logged at step 45500, verifies that the recorded
probabilities agree with the recorded boards, and reports the commit/re-mask
dynamics.

Reproduced (200 samples/split, seed 0, fp32):

| split | this dump | Trainer @45500 |
|---|---|---|
| original | 1.000 | 1.000 |
| extreme_r0 | 0.710 | 0.705 |
| extreme_r1_4 | 0.370 | 0.390 |
| extreme_r5_19 | 0.030 | 0.050 |
| extreme_r20_49 | 0.015 | 0.045 |
| extreme_r50_99 | 0.030 | 0.035 |
| extreme_r100_plus | 0.040 | 0.035 |

**Exact agreement is not expected.** `decoding_strategy: stochastic0.5-linear`
adds Gumbel noise to the remask top-k, so accuracy depends on the RNG stream.
Re-running the hard splits under seeds 1/2/3 gives, for example,
`extreme_r20_49` = 0.025 / 0.055 / 0.030 against 0.015 at seed 0 — a 0.015–0.055
spread that brackets the 0.045 reference. `--dtype float16` reproduces the fp32
numbers exactly, so precision is not a factor. On the near-zero splits these are
counts of 3–11 boards out of 200; treat differences of a few boards as noise.

## Inference steps: more denoising steps buy accuracy

The denoiser **ignores its `t` argument** — `DiffusionModel.forward` accepts `t`
and never uses it, and there is no time embedding anywhere in `src/`. So
`diffusion_steps` can be changed freely at inference; it is not a train/test
conditioning mismatch, it only changes the remask schedule `rate = t/T`.

Sweeping it (200 samples/split, 3 seeds per setting, `--no_paths`):

```bash
for T in 20 40 80 160; do for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$((i++ % 8)) ./.venv/bin/python tools/dump_decoding_paths.py \
    --config configs/sudoku/mdm-extreme-mix10k-train-multidifficulty-eval.yaml \
    --checkpoint output/sudoku/.../checkpoint-45500 \
    --output_dir output/sweep_steps/T${T}_s${s} \
    --diffusion_steps $T --seed $s --no_paths &
done; done; wait
./.venv/bin/python tools/sweep_report.py output/sweep_steps
```

| split | T=20 | T=40 | T=80 | T=160 |
|---|---|---|---|---|
| original | 1.000 | 1.000 | 1.000 | 1.000 |
| extreme_r0 | 0.693 | 0.795 | 0.855 | 0.895 |
| extreme_r1_4 | 0.340 | 0.400 | 0.527 | 0.600 |
| extreme_r5_19 | 0.028 | 0.048 | 0.083 | 0.105 |
| extreme_r20_49 | 0.032 | 0.053 | 0.075 | 0.118 |
| extreme_r50_99 | 0.027 | 0.058 | 0.068 | 0.080 |
| extreme_r100_plus | 0.035 | 0.033 | 0.057 | 0.065 |
| **hard splits pooled** | **0.1925** | **0.2314** | **0.2775** | **0.3106** |

Roughly +3 to +4.6 points pooled per doubling of inference compute, still rising
at T=160 but with the last doubling already returning less than the one before.
The gain is real, not decoder noise: at T=20 vs T=80 the three-seed ranges are
disjoint on `r0`, `r1_4`, `r5_19` and `r50_99`.

`original` is saturated at 1.000 throughout — all of the benefit is on the hard
splits, consistent with the re-masking story above: a finer anneal (T=80 frees
~1 cell of mask budget per step against ~4 at T=20) gives the model far more
opportunities to overturn its own early wrong commits.

## Data format

One JSON per split. `steps[k].xt` is the 81-char board fed to step *k* (`.` =
`[MASK]`), `steps[k].x0` the model's argmax board after it. `probs_b64` is
base64 of a `(steps, 81, 9)` uint8 array — `round(p(digit) * 255)`, indexed
`[(k*81 + cell)*9 + digit]`; `conf_b64` is `(steps, 81)` uint8 of p(argmax) over
the whole vocabulary.

Two consequences of the uint8 quantisation, both benign: the recorded argmax can
disagree with `x0` when two digits round to the same byte (61 cells out of
324 000 on `extreme_r5_19`, all ties — the checker counts these separately), and
probabilities below ~0.2% read as 0. `conf` is over the full 31-token vocabulary
while `probs` covers digits 1–9 only, so their difference is the mass the model
put on non-digit tokens; it is normally under 1%.

Cell *i* of the board is sequence position `82 + i` (81 puzzle characters, then
`[SEP]`, then the 81 solution characters, then `[EOS]`). Position 163 (`[EOS]`)
is also maskable and takes part in the top-k, but it is not a grid cell and is
not recorded.
