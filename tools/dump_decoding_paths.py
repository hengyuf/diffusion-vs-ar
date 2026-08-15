#!/usr/bin/env python
"""Dump MDM decoding paths for every named eval split of a training config.

Reproduces the evaluation the Trainer runs during training -- same splits, same
deterministic sample of each (`eval_num_samples` drawn with
`eval_sample_seed + eval_index`), same denoising loop -- and additionally records,
for every diffusion step, the masked input state, the model's argmax board, and
the full 9-digit probability distribution of all 81 cells.

The generation loop below is a copy of `CustomDiffusionTrainer.generate_samples`
with recording hooks. It must stay numerically identical to it; the only additions
are the `rec.append(...)` blocks.

Output: one JSON per split under --output_dir, plus an index.json. See
tools/README-decoding-viz.md.
"""
import argparse
import base64
import copy
import json
import os
import shutil
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from llmtuner.dsets import get_dataset, preprocess_dataset  # noqa: E402
from llmtuner.tuner.core.custom_tokenizer import CustomTokenizer  # noqa: E402
from llmtuner.tuner.core.parser import get_train_args  # noqa: E402
from llmtuner.tuner.mdm.model import DiffusionModel  # noqa: E402
from llmtuner.tuner.mdm.trainer import topk_decoding  # noqa: E402

N_CELLS = 81
DIGITS = "123456789"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="training YAML the checkpoint came from")
    p.add_argument("--checkpoint", required=True, help="directory holding pytorch_model.bin")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--config_dir", default=None,
                   help="where config.json/tokenizer_config.json live. Defaults to the "
                        "config's model_name_or_path, because a checkpoint-N/ directory "
                        "saves neither.")
    p.add_argument("--num_samples", type=int, default=None,
                   help="override eval_num_samples (default: take it from the config)")
    p.add_argument("--only", default=None,
                   help="comma-separated split aliases to dump (default: all of them). "
                        "Splits keep their original index, so the sample draw is unchanged.")
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                   help="float32 matches training-time eval, which ran on fp32 master "
                        "weights under no autocast.")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the Gumbel noise in stochastic top-k remasking")
    p.add_argument("--diffusion_steps", type=int, default=None,
                   help="override the config's diffusion_steps. Safe to vary at inference: "
                        "the denoiser ignores its `t` argument (no time embedding), so this "
                        "only changes the remask schedule rate = t/T, not model conditioning.")
    p.add_argument("--no_paths", action="store_true",
                   help="write only puzzle/solution/pred/correct, no per-step boards or "
                        "probabilities. For accuracy sweeps, where the ~5 MB/split of path "
                        "data (which scales with diffusion_steps) is not wanted.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(config_dir, checkpoint, diffusion_args, dtype, device):
    """Load the denoiser. Fails loudly if the checkpoint does not actually land."""
    tokenizer = CustomTokenizer.from_pretrained(config_dir)
    config = AutoConfig.from_pretrained(config_dir)
    base = AutoModelForCausalLM.from_config(config)
    model = DiffusionModel(base, config, diffusion_args)

    load_path = os.path.join(checkpoint, "pytorch_model.bin")
    state = torch.load(load_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)

    # The upstream loader swallows load failures with strict=False, which silently
    # evaluates a randomly initialised model. Insist that the real weights arrived.
    loaded = [k for k in state if k.startswith("model.transformer.h.")]
    if not loaded:
        raise RuntimeError(f"{load_path} carries no transformer block weights; keys: {list(state)[:10]}")
    hard_missing = [k for k in missing if k.startswith("model.transformer.h.")]
    if hard_missing:
        raise RuntimeError(f"{len(hard_missing)} transformer weights missing from {load_path}, "
                           f"e.g. {hard_missing[:5]}")
    print(f"[load] {load_path}: {len(state)} tensors, "
          f"{len(missing)} missing / {len(unexpected)} unexpected (shared-module aliases)")

    # DiffusionModel disables the causal mask in __init__; re-apply after the load
    # in case the checkpoint carried a persistent `attn.bias` buffer.
    for block in model.model.transformer.h:
        block.attn.bias.fill_(True)

    model = model.to(getattr(torch, dtype)).to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer


@torch.no_grad()
def generate_with_path(model, tokenizer, diff_args, input_ids, src_mask, digit_ids, device):
    """`generate_samples` from mdm/trainer.py, instrumented.

    Returns (final_xt, records) where each record covers one diffusion step.
    """
    x = input_ids.to(device)
    src_mask = src_mask.bool().to(device)
    attention_mask = torch.ones_like(x)
    batch_size = x.size(0)
    T = diff_args.diffusion_steps

    init_maskable_mask = maskable_mask = ~src_mask
    rec = []

    for t in range(T - 1, -1, -1):
        if t == T - 1:
            xt = x.masked_fill(maskable_mask, tokenizer.mask_token_id)

        xt_in = xt.clone()

        t_tensor = torch.full((batch_size,), t, device=x.device)
        logits = model(xt, t_tensor, attention_mask=attention_mask)
        logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)

        scores = torch.log_softmax(logits.float(), dim=-1)
        scores[:, :, tokenizer.vocab_size:] = -1000
        x0_scores, x0 = scores.max(-1)

        # keep non-[MASK] positions as still
        x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])

        if t > 0:
            xt = topk_decoding(
                x0, x0_scores, diff_args.decoding_strategy,
                init_maskable_mask, t, T, tokenizer.mask_token_id,
            )
        else:
            xt = x0

        rec.append({
            "t": t,
            "xt_in": xt_in,                              # state fed to this forward
            "x0": x0.clone(),                            # argmax board after it
            "conf": x0_scores.exp(),                     # p(argmax), drives the remask top-k
            "digit_probs": scores.exp()[:, :, digit_ids],  # p(1..9) per position
        })

    return xt, rec


def decode_board(tokenizer, ids, id2char):
    return "".join(id2char.get(int(i), "?") for i in ids)


def main():
    args = parse_args()

    args_dict = yaml.safe_load(open(args.config))
    args_dict.update({
        "do_train": False, "do_eval": False, "do_predict": False,
        "report_to": "none", "overwrite_output_dir": True,
        "output_dir": os.path.join(args.output_dir, "_hf_scratch"),
    })
    if args.num_samples is not None:
        args_dict["eval_num_samples"] = args.num_samples
    # The mixture the model trained on is irrelevant here and some of its files are
    # large; only the eval splits matter.
    args_dict["dataset"] = args_dict["eval_dataset"].split(",")[0].split("=")[-1].strip()
    args_dict.pop("max_samples", None)

    model_args, diffusion_args, data_args, training_args, finetuning_args, _ = get_train_args(args_dict)

    if args.diffusion_steps is not None:
        diffusion_args.diffusion_steps = args.diffusion_steps

    config_dir = args.config_dir or model_args.model_name_or_path
    model, tokenizer = load_model(config_dir, args.checkpoint, diffusion_args, args.dtype, args.device)

    vocab = tokenizer.get_vocab()
    digit_ids = [vocab[d] for d in DIGITS]
    id2char = {v: k for k, v in vocab.items() if len(k) == 1}
    pad_id = tokenizer.pad_token_id

    os.makedirs(args.output_dir, exist_ok=True)
    index = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "config": os.path.abspath(args.config),
        "diffusion_steps": diffusion_args.diffusion_steps,
        "decoding_strategy": diffusion_args.decoding_strategy,
        "dtype": args.dtype,
        "seed": args.seed,
        "digits": DIGITS,
        "splits": [],
    }

    named = data_args.get_named_eval_datasets()
    wanted = {s.strip() for s in args.only.split(",")} if args.only else None
    for eval_index, (alias, dataset_name) in enumerate(named):
        # eval_index must stay the split's position in the config: it seeds the sample
        # draw, so filtering must not renumber the splits.
        if wanted is not None and alias not in wanted:
            continue
        # Mirror mdm/workflow.py exactly so we hit the same 200 examples.
        eval_data_args = copy.deepcopy(data_args)
        eval_data_args.dataset = dataset_name
        eval_data_args.eval_dataset = None
        eval_data_args.max_samples = None
        eval_data_args.val_size = 0
        eval_data_args.cache_path = None
        eval_data_args.init_for_training(data_args.eval_sample_seed + eval_index)

        raw = get_dataset(model_args, eval_data_args)
        if data_args.eval_num_samples is not None:
            n = min(len(raw), data_args.eval_num_samples)
            raw = raw.shuffle(seed=data_args.eval_sample_seed + eval_index).select(range(n))
        # keep the untokenised puzzle/solution strings alongside the tensors
        puzzles = list(raw["prompt"])
        solutions = list(raw["response"])

        ds = preprocess_dataset(raw, tokenizer, eval_data_args, training_args, stage="mdm")

        torch.manual_seed(args.seed)
        samples, n_correct = [], 0

        for start in range(0, len(ds), args.batch_size):
            chunk = ds[start:start + args.batch_size]
            input_ids = torch.tensor(chunk["input_ids"])
            src_mask = torch.tensor(chunk["src_mask"])

            src_len = int(src_mask[0].sum())
            assert (src_mask.sum(1) == src_len).all(), "ragged prompts are not supported"
            assert src_len == N_CELLS + 1, f"expected 81 puzzle chars + [SEP], got {src_len}"
            cell_slice = slice(src_len, src_len + N_CELLS)

            final, rec = generate_with_path(
                model, tokenizer, diffusion_args, input_ids, src_mask, digit_ids, args.device)

            src_bool = src_mask.bool().to(final.device)
            pred_ids = final.masked_fill(src_bool, pad_id).cpu()
            label_ids = input_ids.to(final.device).masked_fill(src_bool, pad_id).cpu()

            # per-step arrays: (steps, batch, ...) -> per-sample views below
            xt_all = np.stack([r["xt_in"][:, cell_slice].cpu().numpy() for r in rec])
            x0_all = np.stack([r["x0"][:, cell_slice].cpu().numpy() for r in rec])
            conf_all = np.stack([r["conf"][:, cell_slice].float().cpu().numpy() for r in rec])
            probs_all = np.stack([r["digit_probs"][:, cell_slice].float().cpu().numpy() for r in rec])

            for i in range(input_ids.size(0)):
                gidx = start + i
                # correctness, computed the way core/metric.py:compute_acc does
                dp = tokenizer.decode(pred_ids[i].tolist(), skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)
                dl = tokenizer.decode(label_ids[i].tolist(), skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)
                dp_t, dl_t = dp.strip().split(" "), dl.strip().split(" ")
                correct = dp_t[:len(dl_t)] == dl_t
                n_correct += bool(correct)

                # [MASK] has no single-character form, so it is written as '.'
                steps = [{
                    "t": int(rec[s]["t"]),
                    "xt": "".join("." if int(c) == tokenizer.mask_token_id else id2char.get(int(c), "?")
                                  for c in xt_all[s, i]),
                    "x0": decode_board(tokenizer, x0_all[s, i], id2char),
                } for s in range(len(rec))]

                entry = {
                    "idx": gidx,
                    "puzzle": puzzles[gidx],
                    "solution": solutions[gidx],
                    "pred": decode_board(tokenizer, pred_ids[i][cell_slice].tolist(), id2char),
                    "correct": bool(correct),
                    "n_given": sum(c != "0" for c in puzzles[gidx]),
                }
                if not args.no_paths:
                    probs_u8 = np.round(probs_all[:, i] * 255).astype(np.uint8)   # (T,81,9)
                    conf_u8 = np.round(conf_all[:, i] * 255).astype(np.uint8)     # (T,81)
                    entry["steps"] = steps
                    entry["probs_b64"] = base64.b64encode(probs_u8.tobytes()).decode()
                    entry["conf_b64"] = base64.b64encode(conf_u8.tobytes()).decode()
                samples.append(entry)

        acc = n_correct / len(samples)
        out = {
            "split": alias,
            "dataset": dataset_name,
            "num_samples": len(samples),
            "accuracy": acc,
            "diffusion_steps": diffusion_args.diffusion_steps,
            "digits": DIGITS,
            "samples": samples,
        }
        path = os.path.join(args.output_dir, f"{alias}.json")
        with open(path, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        size_mb = os.path.getsize(path) / 1e6
        print(f"[{alias:18s}] {dataset_name:34s} n={len(samples):4d} acc={acc:.3f}  "
              f"-> {path} ({size_mb:.1f} MB)")

        index["splits"].append({
            "split": alias, "dataset": dataset_name,
            "num_samples": len(samples), "accuracy": acc,
            "n_correct": n_correct, "file": f"{alias}.json",
        })

    with open(os.path.join(args.output_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Drop the viewer in beside the data so output_dir is a servable site on its own.
    # Not for --no_paths dumps: the viewer reads steps/probs_b64, which those omit.
    viewer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decoding_viz", "index.html")
    if os.path.exists(viewer) and not args.no_paths:
        shutil.copy(viewer, os.path.join(args.output_dir, "index.html"))

    print(f"\nWrote index.json with {len(index['splits'])} splits to {args.output_dir}")
    if args.no_paths:
        print("--no_paths: metrics only, no viewer (use tools/sweep_report.py to aggregate)")
    else:
        print(f"View with:  cd {args.output_dir} && python3 -m http.server 8000")


if __name__ == "__main__":
    main()
