"""Legacy compatibility module.

New code should import :class:`KinaSPire` from ``kinaspire.model``.
This module is retained so older model objects serialized from
``kinaspire.kinasenet.KinaseNet`` can still resolve their original path.
"""

from .model import KinaSPire, Encoder, LatentLayer, Decoder

KinaseNet = KinaSPire

__all__ = ["KinaSPire", "KinaseNet", "Encoder", "LatentLayer", "Decoder"]
