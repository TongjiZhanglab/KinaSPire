#!/usr/bin/env python
"""Run the CPTAC tumor-based kinase-activity benchmark.

This script reproduces ``test_tumor_benchmark.ipynb``. Paths are resolved
relative to ``tumor_based_benchmark/``, so it can be launched from any cwd:

    python scripts/run_benchmark.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import benchmark_roc  # noqa: E402

COHORT_LIST = [
    "BIC",
    "CA",
    "CCRCC",
    "GBM",
    "HCC",
    "HNSCC",
    "LUAD",
    "LSCC",
    "OSC",
    "PDA",
    "UCEC",
]

METHOD_DIRS = {
    "KinaSPire": "KinaSPire",
    "z-score": "z-score",
    "mean": "mean",
    "PTM-SEA": "PTM-SEA",
    "PhosX": "PhosX",
    "KSTAR": "KSTAR",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark kinase activity methods on CPTAC tumor cohorts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=BENCH_DIR,
        help="tumor_based_benchmark root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV for subsampled AUROC/AUPRC (default: <root>/results/benchmark_all_result.csv)",
    )
    parser.add_argument("--min-num", type=int, default=30)
    parser.add_argument("--sub-samp-frac", type=float, default=0.8)
    parser.add_argument("--n-pert", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=8376)
    return parser.parse_args(argv)


def load_activity(method_dir: Path, coh: str) -> pd.DataFrame:
    parquet_path = method_dir / f"{coh}.parquet"
    csv_path = method_dir / f"{coh}.csv"
    if parquet_path.is_file():
        df = pd.read_parquet(parquet_path)
    elif csv_path.is_file():
        df = pd.read_csv(csv_path, index_col=0)
    else:
        raise FileNotFoundError(
            f"No activity table for {coh} in {method_dir} "
            f"(looked for {parquet_path.name} and {csv_path.name})"
        )
    df.index.name = None
    return df


def load_inputs(root: Path):
    sparse_path = root / "metadata" / "cohort_sparse_list.pkl"
    with sparse_path.open("rb") as handle:
        sparse_list = pickle.load(handle)

    gs_list = {}
    act_lists = {method: {} for method in METHOD_DIRS}
    label_dir = root / "metadata" / "activating_site_labels"
    activity_root = root / "activity"

    for coh in COHORT_LIST:
        gs_df = pd.read_csv(label_dir / f"{coh}.csv")
        gs_df["is_sparse"] = 0
        gs_df.loc[gs_df["Kinase"].isin(sparse_list[coh]), "is_sparse"] = 1
        gs_list[coh] = gs_df

        for method, dirname in METHOD_DIRS.items():
            act_lists[method][coh] = load_activity(activity_root / dirname, coh)

    return sparse_list, gs_list, act_lists


def run_benchmark(sparse_list, gs_list, act_lists, args):
    frames = []
    for method, act_list in act_lists.items():
        score_lists = {
            coh: act_df[act_df.index.isin(sparse_list[coh])]
            for coh, act_df in act_list.items()
        }
        res = benchmark_roc(
            score_lists,
            gs_list,
            min_num=args.min_num,
            sub_samp_frac=args.sub_samp_frac,
            metric=["auroc", "auprc"],
            n_pert=args.n_pert,
            seed=args.seed,
            return_samples=True,
        )
        res["Method"] = method
        frames.append(res)
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    output = args.output
    if output is None:
        output = root / "results" / "benchmark_all_result.csv"
    else:
        output = output.resolve()

    sparse_list, gs_list, act_lists = load_inputs(root)
    all_result = run_benchmark(sparse_list, gs_list, act_lists, args)

    output.parent.mkdir(parents=True, exist_ok=True)
    all_result.to_csv(output, index=False)
    print(f"Saved benchmark samples to {output}")

    summary = (
        all_result.groupby(["Method", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
