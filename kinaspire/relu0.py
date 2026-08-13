"""Legacy compatibility module for the original activation import path."""

from .activations import ReLU0, ReLUGradAtZero

ReLUgradat0 = ReLUGradAtZero

__all__ = ["ReLU0", "ReLUGradAtZero", "ReLUgradat0"]
