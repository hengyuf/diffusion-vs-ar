#!/usr/bin/env python3
"""Convert downloaded Sudoku corpora to diffusion-vs-ar compatible CSV files."""

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
FIELDS = [
    "quizzes", "solutions", "source", "dataset", "official_rating",
    "rating_type", "difficulty_bucket", "clues", "split",
]


def normalize_puzzle(value: str) -> str:
    value = value.strip().replace(".", "0")
    if len(value) != 81 or any(ch not in "0123456789" for ch in value):
        raise ValueError(f"invalid puzzle: {value[:30]!r}")
    return value


def normalize_solution(value: str) -> str:
    value = value.strip()
    if len(value) != 81 or any(ch not in "123456789" for ch in value):
        raise ValueError(f"invalid solution: {value[:30]!r}")
    return value


def extreme_bucket(rating: float) -> str:
    if rating == 0: return "r0"
    if rating < 5: return "r1_4"
    if rating < 20: return "r5_19"
    if rating < 50: return "r20_49"
    if rating < 100: return "r50_99"
    return "r100_plus"


def kaggle_bucket(rating: float) -> str:
    if rating == 0: return "r0"
    if rating < 1: return "r0_1"
    if rating < 2: return "r1_2"
    if rating < 4: return "r2_4"
    return "r4_plus"


class BucketWriters:
    def __init__(self):
        self.handles = {}
        self.writers = {}
        self.counts = Counter()

    def write(self, name, row):
        if name not in self.writers:
            path = OUT / f"{name}.csv"
            handle = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            self.handles[name], self.writers[name] = handle, writer
        self.writers[name].writerow(row)
        self.counts[name] += 1

    def close(self):
        for handle in self.handles.values(): handle.close()


def convert_extreme(writers):
    for split in ("train", "test"):
        path = RAW / "sudoku_extreme" / f"{split}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rating = float(row["rating"])
                bucket = extreme_bucket(rating)
                puzzle = normalize_puzzle(row["question"])
                writers.write(f"sudoku_extreme_{split}_{bucket}", {
                    "quizzes": puzzle,
                    "solutions": normalize_solution(row["answer"]),
                    "source": row["source"],
                    "dataset": "sudoku_extreme",
                    "official_rating": row["rating"],
                    "rating_type": "tdoku_backtracks",
                    "difficulty_bucket": bucket,
                    "clues": sum(ch != "0" for ch in puzzle),
                    "split": split,
                })


def convert_kaggle(writers):
    path = RAW / "kaggle_3m" / "sudoku-3m.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rating = float(row["difficulty"])
            bucket = kaggle_bucket(rating)
            puzzle = normalize_puzzle(row["puzzle"])
            writers.write(f"sudoku_kaggle3m_train_{bucket}", {
                "quizzes": puzzle,
                "solutions": normalize_solution(row["solution"]),
                "source": "radcliffe_3m",
                "dataset": "sudoku_kaggle3m",
                "official_rating": row["difficulty"],
                "rating_type": "mean_search_depth_10_runs",
                "difficulty_bucket": bucket,
                "clues": row["clues"],
                "split": "train",
            })


def convert_exchange():
    binary = OUT / "solve_sudoku_exchange"
    source = ROOT / "tools" / "solve_sudoku_exchange.cpp"
    subprocess.run(["g++", "-O3", "-std=c++17", str(source), "-o", str(binary)], check=True)
    for bucket in ("easy", "medium", "hard", "diabolical"):
        subprocess.run([
            str(binary), str(RAW / "sudoku_exchange" / "repo" / f"{bucket}.txt"),
            str(OUT / f"sudoku_exchange_train_{bucket}.csv"), "train", bucket,
        ], check=True)
    binary.unlink()


def build_dataset_info():
    info_path = ROOT / "data" / "dataset_info.json"
    existing = {}
    if info_path.exists():
        with info_path.open(encoding="utf-8") as handle: existing = json.load(handle)
    for path in sorted(OUT.glob("sudoku_*.csv")):
        existing[path.stem] = {
            "file_name": str(path.relative_to(ROOT / "data")),
            "columns": {"prompt": "quizzes", "query": "", "response": "solutions", "history": ""},
        }
    with info_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extreme", action="store_true")
    parser.add_argument("--skip-kaggle", action="store_true")
    parser.add_argument("--skip-exchange", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    writers = BucketWriters()
    try:
        if not args.skip_extreme: convert_extreme(writers)
        if not args.skip_kaggle: convert_kaggle(writers)
    finally:
        writers.close()
    if not args.skip_exchange: convert_exchange()
    build_dataset_info()
    manifest = {p.stem: sum(1 for _ in p.open(encoding="utf-8")) - 1 for p in sorted(OUT.glob("sudoku_*.csv"))}
    with (OUT / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
