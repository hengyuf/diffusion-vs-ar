# Training environment on H100 (x86_64)

Companion to [`SETUP-gb200.md`](SETUP-gb200.md). Same `uv` environment, same
pins, same lock file — only the node changed.

Target node: `pool0-01794`, 8x NVIDIA H100 80GB HBM3, `x86_64`, driver
535.216.03 / CUDA 12.2, compute capability 9.0, 128 cores.

## Create the environment

The repo lives on shared lustre, which both clusters mount at the same path:

```bash
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/diffusion-vs-ar
export UV_CACHE_DIR=/lustre/fsw/portfolios/nemotron/users/hengyuf/.cache/uv
export UV_PYTHON_INSTALL_DIR=/lustre/fsw/portfolios/nemotron/users/hengyuf/.local/share/uv/python
uv sync --locked          # ~10 min, first time
```

`uv` downloads CPython 3.11 itself. The venv lands in `.venv/` (7.2 GB).

**Both exports are required on this node.** `$HOME` is a 10 GB NFS volume with
~4 GB free, and uv's default cache (`~/.cache/uv`) plus the managed CPython
would overrun it. Pointing the cache at lustre also puts it on the same
filesystem as `.venv/`, so uv hardlinks instead of copying — the 22 GB of
unpacked wheels under `archive-v0/` and the 7.2 GB `.venv/` are largely the same
blocks, not 29 GB of distinct data.

Verify:

```bash
./.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list(), torch.cuda.device_count())"
# 2.9.1+cu128 ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120'] 8
```

## What changed from the GB200 setup

Nothing in the dependency set. `uv sync --locked` resolved the existing
`uv.lock` with no changes, because the lock is universal and already carries the
`manylinux_2_28_x86_64` torch wheel alongside the `aarch64` one.

Two files had to be recreated: `pyproject.toml` was never committed on the
GB200 branch and was lost with that node's working tree. It is reconstructed
here from `uv.lock`'s `[package.metadata]` block, so it is exact, not
reconstructed by guess. `requirements-h100.txt` mirrors
`requirements-gb200.txt` for non-uv installs.

**cu128 on a CUDA 12.2 driver.** The GB200 node ran driver 580; this one runs
535. That is fine — CUDA minor-version compatibility means any 12.x runtime runs
on a >= 525 driver, and the wheel bundles its own runtime. Verified: fp16 matmul
on device 0, all 8 devices enumerated, NCCL 2.27.5. A `cu126` build would work
equally well; `cu130` would **not** (it needs a 580+ driver), which is why the
lock stays on cu128.

`sm_90` is in the wheel's arch list, so H100 kernels are prebuilt — no JIT.

The `uv run` caveat from `SETUP-gb200.md` was aarch64-specific (the
`nvidia-cusparselt-cu12` tag mismatch does not occur here), but the advice still
holds for a different reason: `uv run` re-syncs on every invocation, and eight
ranks re-syncing into a venv on a shared filesystem can corrupt it. Launch
through the venv interpreter:

```bash
./.venv/bin/accelerate launch --multi_gpu --num_processes 8 ... src/train_bash.py ...
./.venv/bin/python -u src/train_bash.py ...
```

## Running on 8 GPUs

The paper's original recipe is already 8-GPU, so no batch-size surgery is needed
here — the reverse of the GB200 case, where `train-mdm-4gpu.sh` had to double the
per-device batch to 256 to keep the global batch at 1024.

```bash
bash scripts/sudoku/train-mdm-8gpu.sh
```

This is `scripts/sudoku/train-mdm.sh` with the venv interpreter wired in,
`set -euo pipefail`, `mkdir -p`, and a real `--run_name` (the original expands
`${dataset}` before that variable is ever set).

Training still requires a distributed launcher regardless of GPU count:
`src/llmtuner/tuner/mdm/workflow.py` calls `dist.init_process_group()`
unconditionally, so a bare `python src/train_bash.py --do_train` fails with
`environment variable RANK expected`. Use `accelerate launch --num_processes 1`
for single-GPU training. Inference (`--do_predict`) runs fine as a plain
`python` process.

## Verified

All three entry points were run end-to-end on `pool0-01794`, 8 GPUs, fp16,
`model_config_tiny`:

- **MDM training**, `sudoku_train`, 10 steps: NCCL init clean across 8 ranks,
  periodic eval, `pytorch_model.bin` (22 MB) + loss PNGs written.
- **MDM inference** from that checkpoint on `sudoku_test`: 20 denoising steps,
  125 batches, `generated_predictions.jsonl` + `predict_results.json` written.
- **SFT (AR) training + generation**, 10 steps then `--predict_with_generate
  --max_new_tokens 82`: both halves clean.
- **Config-driven multi-difficulty eval**,
  `configs/sudoku/mdm-easy-train-multidifficulty-eval.yaml`: all seven named
  splits (`original`, `extreme_r0` … `extreme_r100_plus`) produced separate
  `eval_*_loss` / `eval_*_acc` keys.

Accuracy from these smoke runs is 0.0 and is meaningless — 6-10 optimizer steps.

## Note on the multi-difficulty config

`configs/sudoku/mdm-easy-train-multidifficulty-eval.yaml` sets
`per_device_train_batch_size: 512` under a comment reading "Eight GPUs give a
global batch size of 8 * 128 = 1024". On 8 GPUs that is a global batch of 4096,
not 1024. This predates the move and was left alone — `learning_rate: 0.002` is
doubled from the paper's 1e-3, which suggests the larger batch was deliberate
and the comment is simply stale. Decide which you want before a real run; the
smoke test above forced 128 only to keep the test short.

## Architecture caveat

`.venv/` now holds x86_64 wheels, and the repo path is shared between the two
clusters. Going back to the GB200 node means `rm -rf .venv && uv sync --locked`
to repopulate it for aarch64 — the lock covers both, but a venv does not. If you
need both live at once, set `UV_PROJECT_ENVIRONMENT=.venv-$(uname -m)` and point
the scripts' `VENV=` line at it.

## Data

`data/sudoku_train.csv`, `data/sudoku_test.csv`, and all of
`data/processed/*.csv` are present on lustre — nothing to download.
