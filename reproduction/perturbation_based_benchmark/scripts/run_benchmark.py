#!/usr/bin/env python
"""Run the perturbation-based kinase-activity benchmark.

This script reproduces ``test_perturb_benchmark.ipynb``. Paths are resolved
relative to ``perturbation_based_benchmark/``, so it can be launched from any cwd:

    python scripts/run_benchmark.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils import compute_recall_at_k, run_perturb_bench, run_phit, run_rank  # noqa: E402

METHODS = [
    "KinaSPire",
    "PhosX",
    "KSTAR",
    "z-score",
    "PTM-SEA",
    "mean",
]

RANGES = [
    ("<5", lambda counts: counts < 5),
    ("≥5", lambda counts: counts >= 5),
    ("All", lambda counts: pd.Series(True, index=counts.index)),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark kinase activity methods on perturbation experiments."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=BENCH_DIR,
        help="perturbation_based_benchmark root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pickle for full results (default: <root>/results/benchmark_all_result.pkl)",
    )
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=float, nargs="+", default=[0.1, 0.2])
    return parser.parse_args(argv)


def load_activity(activity_dir: Path, method: str) -> pd.DataFrame:
    path = activity_dir / f"{method}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing activity table: {path}")
    df = pd.read_csv(path, index_col=0)
    df.index.name = None
    return df


def sign_unify(act: pd.DataFrame, sign_series: pd.Series) -> pd.DataFrame:
    signed = act.copy()
    common_cols = signed.columns.intersection(sign_series.index)
    signed[common_cols] = signed[common_cols] * sign_series[common_cols]
    return signed


def load_inputs(root: Path):
    curated = pd.read_csv(root / "metadata" / "curated_sub_count.csv", index_col=0)
    meta = pd.read_csv(root / "metadata" / "meta.csv", index_col=0)
    meta_sign = meta[["id", "sign"]].drop_duplicates()
    sign_series = meta_sign.set_index("id")["sign"]

    acts = {}
    signed_acts = {}
    activity_dir = root / "activity"
    for method in METHODS:
        act = load_activity(activity_dir, method)
        acts[method] = act
        signed_acts[method] = sign_unify(act, sign_series)
    return curated, meta, acts, signed_acts


def filter_activity(act: pd.DataFrame, kinases) -> pd.DataFrame:
    filtered = act[act.index.isin(kinases)]
    return filtered.loc[~filtered.isna().all(axis=1), ~filtered.isna().all()]


def evaluate_method(act, act_signed, meta, curated_subset, topks, n_iter, seed):
    row = {}
    act_signed = filter_activity(act_signed, curated_subset.index)
    act = filter_activity(act, curated_subset.index)

    row["n_Detected"] = len(set(act.index) & set(curated_subset["source"]))
    row["n_Overlap"] = len(
        set(act.index) & set(curated_subset["source"]) & set(meta["target"])
    )

    if row["n_Overlap"] == 0:
        for topk in topks:
            row[f"Phit@{topk}"] = np.nan
            row[f"macroRecall@{topk}"] = np.nan
        row["Rank (paper)"] = np.nan
        row["AUROC"] = np.nan
        return row

    for topk in topks:
        row[f"macroRecall@{topk}"] = float(
            compute_recall_at_k(
                meta, act_signed, k=topk, mode="macro", metric="scaled_rank"
            )["recall_at_k"]
        )
        row[f"Phit@{topk}"] = float(
            run_phit(act_signed, meta, k=topk, average=True, metric="scaled_rank")
        )

    rank_tbl = run_rank(act_signed, meta, average=True)
    row["Rank (paper)"] = rank_tbl["scaled_rank"].tolist() if len(rank_tbl) > 0 else np.nan
    row["n_Overlap"] = len(rank_tbl)

    if row["n_Overlap"] > 0:
        out = run_perturb_bench(
            act, meta, scale_data=True, rm_bg=False, n_iter=n_iter, random_state=seed
        )
        row["AUROC"] = list(out) if not isinstance(out, list) else out
    else:
        row["AUROC"] = np.nan
    return row


def print_range_summary(range_name, res_df):
    print(f"\n# {range_name}")
    for _, row in res_df.iterrows():
        rank = row["Rank (paper)"]
        auroc = row["AUROC"]
        mean_rank = np.nan if (isinstance(rank, float) and np.isnan(rank)) else np.mean(rank)
        med_auroc = np.nan if (isinstance(auroc, float) and np.isnan(auroc)) else np.median(auroc)
        print(row["method"], mean_rank, med_auroc)


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    output = args.output
    if output is None:
        output = root / "results" / "benchmark_all_result.pkl"
    else:
        output = output.resolve()

    curated, meta, acts, signed_acts = load_inputs(root)
    frames = []

    for range_name, predicate in RANGES:
        curated_subset = curated[predicate(curated["count"])].copy()
        results = {}
        for method in METHODS:
            results[method] = evaluate_method(
                acts[method],
                signed_acts[method],
                meta,
                curated_subset,
                args.topk,
                args.n_iter,
                args.seed,
            )
        res_df = pd.DataFrame.from_dict(results, orient="index")
        res_df = res_df.reset_index().rename(columns={"index": "method"})
        res_df["n_Total"] = len(curated_subset)
        res_df["range"] = range_name
        print_range_summary(range_name, res_df)
        frames.append(res_df)

    all_res = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(all_res, handle)

    summary = all_res.drop(columns=["Rank (paper)", "AUROC"], errors="ignore").copy()
    summary["mean_rank_paper"] = all_res["Rank (paper)"].apply(
        lambda x: np.nan if (isinstance(x, float) and np.isnan(x)) else float(np.mean(x))
    )
    summary["median_auroc"] = all_res["AUROC"].apply(
        lambda x: np.nan if (isinstance(x, float) and np.isnan(x)) else float(np.median(x))
    )
    summary_path = output.with_suffix(".summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved full results to {output}")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
