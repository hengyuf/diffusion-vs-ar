# Training environment on GB200 (aarch64)

The `Setup` section of the main README (conda + python 3.9 + `torch==1.13.1+cu117`)
cannot be used on this cluster: there are no arm64 wheels for that stack, and cu117
has no Blackwell (`sm_100`) kernels. This document describes the `uv` environment
that replaces it.

Target node: 4x NVIDIA GB200, `aarch64`, driver 580 / CUDA 13.0, compute capability 10.0.

## Create the environment

```bash
cd /lustre/fsw/portfolios/nemotron/users/hengyuf/diffusion-vs-ar
uv sync            # ~1 min, resolves pyproject.toml -> uv.lock -> .venv/
```

That is all — `uv` downloads CPython 3.11 itself. The venv lands in `.venv/` (6.9 GB)
and is checked against `uv.lock`, so it reproduces exactly on any node of this cluster.

Verify:

```bash
./.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list(), torch.cuda.device_count())"
# 2.9.1+cu128 ['sm_80', 'sm_90', 'sm_100', 'sm_120'] 4
```

## What changed from `requirements.txt`, and why

| package | original | here | reason |
|---|---|---|---|
| python | 3.9 | 3.11 | no aarch64 wheels for this stack on 3.9; `datasets` 2.16 breaks on 3.12 |
| torch | 1.13.1+cu117 | 2.9.1+cu128 | newest release with `manylinux_2_28_aarch64` wheels carrying `sm_100` kernels |
| datasets | 2.14.7 | 2.16.1 | same API, tolerates pyarrow >= 14 |
| wandb | 0.14.0 | 0.18.7 | 0.14 predates aarch64 wheels |
| numpy | (unpinned) | `<2` | transformers 4.37 / datasets 2.16 predate the numpy 2 ABI |
| transformers, accelerate, peft | 4.37.2 / 0.27.2 / 0.6.0 | unchanged | see below |

**Do not bump transformers.** The `mdm` and `sft` trainers subclass `Trainer`
internals whose signatures changed after 4.37, and every script in `scripts/`
passes `--evaluation_strategy`, which was removed in transformers 4.46.

`torch` resolves from `https://download.pytorch.org/whl/cu128` (declared as an
explicit index in `pyproject.toml`); everything else comes from PyPI. The cu128
runtime is forward-compatible with the node's CUDA 13.0 driver — a `cu130` build
exists but buys nothing here.

`requirements-gb200.txt` mirrors these pins for anyone installing with plain pip.

## Running

Launch through the venv interpreter, **not** `uv run`:

```bash
./.venv/bin/accelerate launch --multi_gpu --num_processes 4 ... src/train_bash.py ...
./.venv/bin/python -u src/train_bash.py ...
```

`uv run` re-installs `nvidia-cusparselt-cu12` on *every* invocation — NVIDIA's
aarch64 wheel is tagged `manylinux2014_sbsa` internally while its filename says
`aarch64`, so uv reads the installed copy as a tag mismatch. It is only ~100 ms,
but several ranks re-installing into a venv on a shared filesystem is a way to
corrupt it. If you want `uv run` anyway, use `uv run --no-sync` (or export
`UV_NO_SYNC=1`).

A ready 4-GPU version of the 8-GPU sudoku recipe, with the per-device batch size
raised 128 -> 256 to keep the paper's global batch of 1024:

```bash
bash scripts/sudoku/train-mdm-4gpu.sh
```

Note that training requires a distributed launcher regardless of GPU count:
`src/llmtuner/tuner/mdm/workflow.py` calls `dist.init_process_group()`
unconditionally, so a bare `python src/train_bash.py --do_train` fails with
`environment variable RANK expected`. Use `accelerate launch --num_processes 1`
for single-GPU training. Inference (`--do_predict`) runs fine as a plain
`python` process.

`--fp16` is kept to match the paper. bf16 is the more natural precision on
Blackwell; `--bf16` in place of `--fp16` (and `--mixed_precision bf16`) also works.

## Verified

Both halves of the workflow were run end-to-end on `nvl72d057-T11`:

- `mdm` training, 4 GPUs, fp16, `sudoku_train`, `model_config_tiny`: 10 steps,
  NCCL init clean, loss 6.24 -> 3.9, periodic eval, checkpoint written.
- `mdm` inference from that checkpoint on `sudoku_test`: 20 denoising steps,
  `generated_predictions.jsonl` + `predict_results.json` written. (torch 2.6
  flipped the `torch.load` `weights_only` default; the `pytorch_model.bin`
  reload in `tuner/core/loader.py` is unaffected because it stores plain tensors.)

Accuracy from those smoke runs is 0.0 and is meaningless — 10 optimizer steps.

## Data

`data/sudoku_train.csv` and `data/sudoku_test.csv` are present. The
difficulty-stratified corpora under `data/processed/` are **not** downloaded yet
(only `manifest.json` is there), so the `sudoku_extreme_*` / `sudoku_exchange_*`
entries in `data/dataset_info.json` will fail until you fetch them. The `hf` CLI
needs a much newer `huggingface_hub` than transformers 4.37 tolerates, so run it
outside the training env:

```bash
uvx --from "huggingface_hub[hf_transfer,cli]" hf download \
  fhyfhy/diffusion-vs-ar-hard-sudoku --repo-type dataset --local-dir data \
  --include "processed/*.csv" "processed/manifest.json"
```
