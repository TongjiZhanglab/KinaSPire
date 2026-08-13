import argparse

from kinaspire.preprocessing import DataProcessor


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess phosphoproteomics data and a KSR prior for KinaSPire."
    )
    parser.add_argument("--exp", required=True, help="Phosphosite matrix (sites × samples), .feather")
    parser.add_argument("--ksr", required=True, help="KSR prior (sites × kinases), .tsv")
    parser.add_argument("--output", required=True, help="Run directory; writes data.parquet and prior.parquet under processed_data/")
    parser.add_argument("--with-centering", action="store_true", help="Center features before scaling")
    parser.add_argument(
        "--quantile-range",
        nargs=2,
        type=float,
        default=(1, 99),
        metavar=("QMIN", "QMAX"),
        help="Robust scaler quantile range (default: 1 99)",
    )
    parser.add_argument(
        "--unit-variance",
        action="store_true",
        help="Scale features to unit variance after robust scaling",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    processor = DataProcessor(
        exp_path=args.exp,
        ksr_path=args.ksr,
        output_path=args.output,
        with_centering=args.with_centering,
        quantile_range=tuple(args.quantile_range),
        unit_variance=args.unit_variance,
    )
    processor.process_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
