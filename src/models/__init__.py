"""Model registry for Deoghar Forest Cover Analysis.

Lazy imports so the baselines (no PyTorch) can be loaded without torch installed.
"""
from __future__ import annotations

from .baselines import CAMarkov, LinearExtrapolation, RandomForestBaseline

_TORCH_MODELS = ("convlstm", "unet_convlstm")


def build_model(name: str, **kwargs):
    """Build a model by name. kwargs are passed to the constructor."""
    name = name.lower()
    if name == "convlstm":
        from .convlstm import ConvLSTM
        return ConvLSTM(**kwargs)
    if name == "unet_convlstm":
        from .convlstm import UNetConvLSTM
        return UNetConvLSTM(**kwargs)
    raise ValueError(f"Unknown model '{name}'. Available torch models: {list(_TORCH_MODELS)}")


def __getattr__(name):
    if name in ("ConvLSTM", "ConvLSTMCell", "UNetConvLSTM"):
        from . import convlstm
        return getattr(convlstm, name)
    raise AttributeError(name)


__all__ = [
    "ConvLSTM", "ConvLSTMCell", "UNetConvLSTM",
    "CAMarkov", "LinearExtrapolation", "RandomForestBaseline",
    "build_model",
]
