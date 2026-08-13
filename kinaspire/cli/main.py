import argparse
import sys
from importlib.metadata import PackageNotFoundError, version


def _package_version():
    try:
        return version("kinaspire")
    except PackageNotFoundError:
        from kinaspire import __version__
        return __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="kinaspire",
        description="KinaSPire: a prior-constrained framework for sample-specific kinase activity inference under incomplete substrate knowledge.",
        epilog=(
            "commands:\n"
            "  preprocess  Preprocess phosphoproteomics data and a KSR prior\n"
            "  train       Train a KinaSPire model\n"
            "  infer       Infer scaled kinase activity\n"
            "\n"
            "Run `kinaspire <command> -h` for command-specific help."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"kinaspire {_package_version()}",
    )
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    commands = {"preprocess", "train", "infer"}
    if argv and argv[0] in commands:
        command, rest = argv[0], argv[1:]
        if command == "preprocess":
            from .preprocess import main as preprocess_main
            return preprocess_main(rest)
        if command == "train":
            from .train import main as train_main
            return train_main(rest)
        from .infer import main as infer_main
        return infer_main(rest)

    if not argv:
        parser.print_help()
        return 0

    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
