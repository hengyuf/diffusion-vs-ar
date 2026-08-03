#!/usr/bin/env bash
# train-mdm.sh adapted for a single 4-GPU GB200 node (see SETUP-gb200.md).
#
# Differences from the 8-GPU original:
#   --num_processes 4 and per_device_train_batch_size 256, which keeps the
#   paper's global batch of 1024 (8 x 128).
#
# Launch through the venv interpreter, not `uv run`: uv re-installs
# nvidia-cusparselt-cu12 on every invocation (its aarch64 wheel is tagged
# manylinux2014_sbsa internally), and concurrent ranks re-installing into a
# shared-filesystem venv is a good way to corrupt it.
set -euo pipefail

cd "$(dirname "$0")/../.."
VENV="$PWD/.venv/bin"

export WANDB_DISABLED=true

exp=output/sudoku/mdm-4gpu-alpha0.25-gamma1-bs1024-lr1e-3-ep300-T20-$(date "+%Y%m%d-%H%M%S")
mkdir -p "$exp"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
"$VENV/accelerate" launch --multi_gpu --num_machines 1 --mixed_precision fp16 \
  --num_processes 4 --main_process_port 20099 \
  src/train_bash.py \
    --stage mdm --overwrite_output_dir \
    --cache_dir ./cache \
    --model_name_or_path model_config_tiny \
    --do_train \
    --dataset sudoku_train \
    --finetuning_type full \
    --cutoff_len 164 \
    --output_dir "$exp" \
    --overwrite_cache \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --val_size 448 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --num_train_epochs 300.0 \
    --plot_loss \
    --run_name sudoku-mdm-4gpu \
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
    --gamma 1 \
  > "$exp/train.log" 2>&1

for dataset in sudoku_test; do
  topk_decoding=True
  mkdir -p "$exp/$dataset"
  CUDA_VISIBLE_DEVICES=0 \
  "$VENV/python" -u src/train_bash.py \
      --stage mdm --overwrite_output_dir \
      --cache_dir ./cache \
      --model_name_or_path model_config_tiny \
      --do_predict \
      --cutoff_len 164 \
      --dataset "$dataset" \
      --finetuning_type full \
      --diffusion_steps 20 \
      --output_dir "$exp/$dataset" \
      --checkpoint_dir "$exp" \
      --remove_unused_columns False \
      --decoding_strategy stochastic0.5-linear \
      --topk_decoding $topk_decoding \
    > "$exp/$dataset/eval-TopK$topk_decoding.log" 2>&1
done
