# Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning


This repository contains code for training and evaluating the models in the paper [Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning](https://arxiv.org/abs/2410.14157).

<p align = "center">
<img src="simple_task.png" width="95%" alt="simple_task" align=center />
</p>

- Autoregressive language models, despite their impressive capabilities, struggle with complex reasoning and long-term planning tasks. Can we go beyond autoregression for these challanges?
- First, what is planning essentially? We design a straightforward task to minimally illustrate planning, where we can also control the extent of planning through a term called Planning Distance. We find AR struggles a lot on this simple task. 
- Then, we delve into the comparison between the objective of autoregression and discrete diffusion, and demonstrate how discrete diffusion models effectively learn difficult subgoals that elude autoregressive models. 
- Based on above, we further introduce Multi-granularity Diffusion Modeling (MDM), which prioritizes subgoals based on difficulty during learning. We find MDM significantly outperforms AR on various more complex reasoning and planning challanges.


## Setup
All required packages can be found in requirements.txt. You can install them in a new environment with
```
conda create -n diffusion python=3.9
conda activate diffusion
git clone git@github.com:HKUNLP/diffusion-vs-ar.git

cd diffusion-vs-ar
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage
Training and evaluation commands are provided under the `scripts` directory. Download data from [here](https://drive.google.com/file/d/1b0OIlYL76rVVuNYIfIb-L_Ptdg6k_y0c/view?usp=sharing) first.
The synthetic planning dataset can also be generated using `data/synthetic_graph.py` script.
```
# run AR (training from scratch)
bash scripts/sudoku/train-sft.sh

# run AR (finetuning from LLaMA)
bash scripts/sudoku/train-sft-llama-7b.sh

# run Diffusion (training from scratch)
bash scripts/sudoku/train-mdm.sh
```
(📌check our work on scaling diffusion langauge model by adapting from LLaMA at https://github.com/HKUNLP/DiffuLLaMA)

For experiment on different model size, change `--model_name_or_path` to `model_config_tiny` (~6M), `model_config` (~85M) or `model_config_medium` (~303M). A slightly larger learning rate (i.e., 1e-3) is used for the tiny model.

For experiment on different datasets, change `--dataset` (dataset name in `data/dataset_info.json`) and adjust `--cutoff_len` (make sure equal or larger than the largest token length on that dataset). For AR, make sure the `--max_new_tokens` during generation is also larger than that seen in the training time. Here are the `cutoff_len` and `--max_new_tokens` used in the paper:

|                      | Minimal Planning | Countdown 3 | Countdown 4 | Countdown 5 | Sudoku | 3-SAT 5v | 3-SAT 7v | 3-SAT 9v |
|----------------------|:----------------:|:-----------:|:-----------:|:-----------:|:------:|:--------:|:--------:|:--------:|
| cutoff_len           |               75 |      37     |      64     |          74 |    164 |      258 |      285 |      325 |
| max_new_tokens (sft) | 24               |          24 |          32 |          54 |     82 |       10 |       14 |       18 |

Please refer to Appendix C.2 for other illustraion of implementation details.

## Hard Sudoku datasets

The original Sudoku data is intentionally easy: sampled puzzles can be solved
with naked-single propagation and require no search. This branch adds
difficulty-stratified Sudoku corpora for experiments involving candidate
assumptions, search, and backtracking. The converted data is hosted at
[`fhyfhy/diffusion-vs-ar-hard-sudoku`](https://huggingface.co/datasets/fhyfhy/diffusion-vs-ar-hard-sudoku).

### Download

Install the Hugging Face CLI and download directly into the layout expected by
this repository:

```bash
pip install -U huggingface_hub
hf download fhyfhy/diffusion-vs-ar-hard-sudoku \
  --repo-type dataset \
  --local-dir data \
  --include "processed/*.csv" "processed/manifest.json" \
            "sudoku_train.csv" "sudoku_test.csv"
```

After downloading, the important paths are:

```text
data/dataset_info.json
data/sudoku_train.csv
data/sudoku_test.csv
data/processed/sudoku_extreme_train_r50_99.csv
data/processed/sudoku_extreme_test_r50_99.csv
...
```

All processed files begin with the original compatible columns
`quizzes,solutions`. They additionally preserve `source`, `official_rating`,
`rating_type`, `difficulty_bucket`, `clues`, and `split`. Exact counts are in
[`data/processed/manifest.json`](data/processed/manifest.json).

Difficulty systems are kept separate:

| family | rating | buckets |
|---|---|---|
| Sudoku Extreme | tdoku backtracks | `r0`, `r1_4`, `r5_19`, `r20_49`, `r50_99`, `r100_plus` |
| Kaggle 3M | mean search depth over 10 runs | `r0`, `r1_2`, `r2_4`, `r4_plus` |
| Sudoku Exchange | Sukaku Explainer | `easy`, `medium`, `hard`, `diabolical` |

Do not compare raw rating numbers across families: each rating is produced by a
different solver and measures a different notion of difficulty.

### Train on one difficulty bucket

The following is the original 8-GPU MDM setup changed to train on puzzles with
50--99 tdoku backtracks:

```bash
exp=output/sudoku/mdm-extreme-r50-99-$(date "+%Y%m%d-%H%M%S")
mkdir -p "$exp"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --multi_gpu --num_machines 1 --mixed_precision fp16 \
  --num_processes 8 --main_process_port 20099 src/train_bash.py \
  --stage mdm --overwrite_output_dir \
  --cache_dir ./cache \
  --model_name_or_path model_config_tiny \
  --do_train \
  --dataset sudoku_extreme_train_r50_99 \
  --finetuning_type full \
  --cutoff_len 164 \
  --output_dir "$exp" \
  --overwrite_cache \
  --per_device_train_batch_size 128 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --val_size 448 \
  --per_device_eval_batch_size 32 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_steps 500 \
  --learning_rate 1e-3 \
  --num_train_epochs 300 \
  --plot_loss \
  --run_name sudoku-extreme-r50-99 \
  --preprocessing_num_workers 8 \
  --fp16 \
  --save_total_limit 1 \
  --remove_unused_columns False \
  --diffusion_steps 20 \
  --save_safetensors False \
  --token_reweighting True \
  --time_reweighting linear \
  --topk_decoding True \
  --alpha 0.25 \
  --gamma 1
```

For the 85M or 303M configurations, replace `model_config_tiny` with
`model_config` or `model_config_medium` and retune the learning rate and batch
size.

### Train with a difficulty mixture

The loader supports probabilistic interleaving. For a planning-heavy curriculum:

```bash
--dataset sudoku_extreme_train_r5_19,sudoku_extreme_train_r20_49,sudoku_extreme_train_r50_99,sudoku_extreme_train_r100_plus \
--mix_strategy interleave_over \
--interleave_probs 0.1,0.3,0.4,0.2
```

Add these arguments to the training command above in place of its single
`--dataset` argument. `interleave_over` continues until all constituent data
have been consumed; use `interleave_under` to stop when the first constituent
is exhausted. For reproducible comparisons, report both the mixture weights and
the number of optimizer steps.

### Evaluate by difficulty

Evaluate each bucket separately so easy examples do not hide failure on hard
ones:

```bash
checkpoint=output/sudoku/YOUR_RUN

for bucket in r5_19 r20_49 r50_99 r100_plus; do
  out="$checkpoint/eval_$bucket"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 python3 -u src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --checkpoint_dir "$checkpoint" \
    --do_predict \
    --dataset "sudoku_extreme_test_$bucket" \
    --finetuning_type full \
    --cutoff_len 164 \
    --diffusion_steps 20 \
    --output_dir "$out" \
    --remove_unused_columns False \
    --decoding_strategy stochastic0.5-linear \
    --topk_decoding True \
    > "$out/eval.log"
done
```

Each output directory contains `generated_predictions.jsonl` and
`predict_results.json`. The reported Sudoku accuracy is whole-board exact match,
not per-cell accuracy.

### Weights & Biases

The original scripts set `WANDB_DISABLED=true`. To log both training loss and
periodic `eval_loss`/`eval_acc`, remove that line and use:

```bash
unset WANDB_DISABLED
export WANDB_PROJECT=diffusion-vs-ar-hard-sudoku
wandb login
```

Then add `--report_to wandb --run_name YOUR_RUN_NAME` to the training command.

### Rebuild the converted files

The reproducible streaming converter and Sudoku Exchange solution generator are
in `tools/`:

```bash
python3 tools/prepare_sudoku_datasets.py
```

The converter preserves source ratings, produces physical bucket files that the
existing loader can mix without code changes, and regenerates
`data/dataset_info.json` and the manifest.

## Citation
If you find our code or data helpful, please cite us as follows 
```
@article{ye2024beyond,
  title={Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning},
  author={Ye, Jiacheng and Gao, Jiahui and Gong, Shansan and Zheng, Lin and Jiang, Xin and Li, Zhenguo and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2410.14157},
  year={2024}
}
```

The code framework is adapted from [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory), thanks for their great work.
