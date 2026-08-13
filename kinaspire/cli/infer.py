import argparse
from pathlib import Path

import pandas as pd
import torch

from kinaspire.activity import cal_kin_act, scale_kin_act, split_cols
from kinaspire.cli._io import load_table, save_table


def build_parser():
    parser = argparse.ArgumentParser(
        description="Infer scaled kinase activity from a trained KinaSPire model."
    )
    parser.add_argument("--data", required=True, help="Preprocessed data table (samples × sites)")
    parser.add_argument("--prior", required=True, help="Preprocessed prior table (kinases × sites)")
    parser.add_argument("--model", default=None, help="Path to a saved .pth model")
    parser.add_argument(
        "--performance",
        default=None,
        help="performance.csv used to pick the best model by val_r2 when --model is omitted",
    )
    parser.add_argument("--output", required=True, help="Output table for scaled kinase activity")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument(
        "--meta-kin",
        action="store_true",
        help="Use the kinase–kinase layer when extracting activity",
    )
    return parser


def _select_model(performance_path):
    perf = pd.read_csv(performance_path).sort_values("val_r2", ascending=False)
    if perf.empty:
        raise ValueError(f"No rows in performance file: {performance_path}")
    best = perf.iloc[0]
    cv = int(best["cv"])
    wd = best["weight_decay"]
    refit = best["refit"]
    refit_tag = "None" if pd.isna(refit) else str(int(refit))
    model_dir = Path(performance_path).parent / "model"
    model_path = model_dir / f"cv{cv}_wd{wd}_refit{refit_tag}.pth"
    if not model_path.is_file():
        raise FileNotFoundError(f"Best model not found: {model_path}")
    return model_path


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.model is None and args.performance is None:
        raise SystemExit("Provide --model or --performance to select a trained model.")

    model_path = Path(args.model) if args.model else _select_model(args.performance)
    print(f"Using model: {model_path}")

    data = load_table(args.data)
    prior = load_table(args.prior)
    model = torch.load(model_path, weights_only=False, map_location=args.device)

    kin_act = cal_kin_act(
        data=data,
        prior=prior,
        model=model,
        batch_size=args.batch_size,
        meta_kin=args.meta_kin,
        device=args.device,
    )
    kin_act = split_cols(kin_act).T
    kin_act_scaled = scale_kin_act(kin_act, prior)
    save_table(kin_act_scaled, args.output)
    print(f"Saved scaled kinase activity to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
