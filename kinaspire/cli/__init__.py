"""Command-line entry points for KinaSPire."""

from .preprocess import main as preprocess_main
from .train import main as train_main
from .infer import main as infer_main

__all__ = ["preprocess_main", "train_main", "infer_main"]
