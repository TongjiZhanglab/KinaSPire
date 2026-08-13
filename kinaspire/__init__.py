"""KinaSPire core package."""

from .model import KinaSPire, Encoder, LatentLayer, Decoder
from .activations import ReLU0
from .config import load_config, save_config
from .data import DataSplitter
from .preprocessing import RobustMinScaler, DataProcessor
from .activity import cal_kin_act, scale_kin_act

__all__ = [
    "KinaSPire",
    "Encoder",
    "LatentLayer",
    "Decoder",
    "ReLU0",
    "DataSplitter",
    "RobustMinScaler",
    "DataProcessor",
    "Trainer",
    "OptimizerSetter",
    "Evaluator",
    "fit_model",
    "refit_model",
    "load_config",
    "save_config",
    "cal_kin_act",
    "scale_kin_act",
]

__version__ = "0.1.0"


def __getattr__(name):
    if name in {"Trainer", "OptimizerSetter", "fit_model", "refit_model"}:
        from .training import Trainer, OptimizerSetter, fit_model, refit_model
        return {
            "Trainer": Trainer,
            "OptimizerSetter": OptimizerSetter,
            "fit_model": fit_model,
            "refit_model": refit_model,
        }[name]
    if name == "Evaluator":
        from .evaluation import Evaluator
        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
