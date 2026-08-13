import argparse

from kinaspire.cli._io import load_table
from kinaspire.training import fit_model


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a KinaSPire model from preprocessed data and a YAML config."
    )
    parser.add_argument("--data", required=True, help="Preprocessed data table (samples × sites)")
    parser.add_argument("--prior", required=True, help="Preprocessed prior table (kinases × sites)")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--output", required=True, help="Directory for models and performance.csv")
    parser.add_argument("--ppi", default=None, help="Optional kinase–kinase mask table")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data = load_table(args.data)
    prior = load_table(args.prior)
    ppi_mask = load_table(args.ppi) if args.ppi else None
    fit_model(
        data,
        prior,
        config=args.config,
        ppi_mask=ppi_mask,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
