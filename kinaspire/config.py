"""Load nested YAML training configs for :func:`kinaspire.training.fit_model`."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


# PyYAML implements YAML 1.1, which treats `1e-3` as a string and only
# accepts scientific notation with a decimal point (`1.0e-3`). Add the
# YAML 1.2 float forms so configs can use either spelling.
class _ConfigLoader(yaml.SafeLoader):
    pass


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                    |[-+]?\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?\.(?:inf|Inf|INF)
                    |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


DEFAULT_CONFIG = {
    "data": {
        "val_size": 0.3,
        "batch_size": 32,
        "fraction_gs": 0.0,
    },
    "model": {
        "dropout_prior": 0.05,
        "dropout_meta": 0.0,
        "dropout_decoder": 0.0,
    },
    "training": {
        "num_epochs": 200,
        "cvs": 5,
    },
    "refit": {
        "num_epochs": 50,
        "iterations": 5,
        "resample": True,
    },
    "optimizer": {
        "name": "Adam",
        "learning_rate": 1.0e-4,
        "weight_decays": (-10, -1, 4),
        "kwargs": {},
        "parameter_kwargs": {"prior": {"weight_decay": 1e-10}},
    },
    "scheduler": {
        "name": "CosineAnnealingLR",
        "kwargs": {"T_max": 10},
    },
    "pruning": {
        "eps_factor": 10,
        "fill_zeroed": True,
    },
    "runtime": {
        "device": "cpu",
    },
}


def _merge_section(base, override):
    merged = copy.deepcopy(base)
    if not override:
        return merged
    for section, values in override.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section].update(copy.deepcopy(values))
        else:
            merged[section] = copy.deepcopy(values)
    return merged


def load_config(source=None):
    """Load a nested config from a YAML path, a dict, or package defaults.

    Missing keys are filled from :data:`DEFAULT_CONFIG`. Keys present in the
    YAML file replace the corresponding default, including empty dicts.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if source is None:
        return cfg
    if isinstance(source, dict):
        return _merge_section(cfg, source)
    if isinstance(source, (str, os.PathLike, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.load(handle, Loader=_ConfigLoader) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"Config file must contain a mapping, got {type(loaded).__name__}")
        return _merge_section(cfg, loaded)
    raise TypeError(f"config must be a YAML path or dict, got {type(source).__name__}")


def save_config(cfg, path):
    """Write a nested config dict to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_cfg = copy.deepcopy(cfg)
    weight_decays = dump_cfg.get("optimizer", {}).get("weight_decays")
    if isinstance(weight_decays, tuple):
        dump_cfg["optimizer"]["weight_decays"] = list(weight_decays)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dump_cfg, handle, sort_keys=False, allow_unicode=True)


def _resolve_torch_class(module, name, kind):
    if isinstance(name, type):
        return name
    if not isinstance(name, str) or not name:
        raise TypeError(f"{kind} name must be a class or non-empty string, got {name!r}")
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown {kind}: {name}") from exc


def as_fit_params(cfg):
    """Flatten a nested config into attributes used by the training loop."""
    optimizer = cfg["optimizer"]
    scheduler = cfg["scheduler"]
    pruning = cfg["pruning"]

    weight_decays = optimizer["weight_decays"]
    if optimizer.get("weight_decays_logspace") or isinstance(weight_decays, tuple):
        weight_decays = np.logspace(*weight_decays)

    eps = pruning.get("eps")
    if eps is None:
        eps = torch.finfo(torch.float).eps

    return SimpleNamespace(
        data_val_size=cfg["data"]["val_size"],
        batch_size=cfg["data"]["batch_size"],
        fraction_gs=cfg["data"]["fraction_gs"],
        dropout_rate1=cfg["model"]["dropout_prior"],
        dropout_rate2=cfg["model"]["dropout_meta"],
        dropout_rate3=cfg["model"]["dropout_decoder"],
        num_epochs=cfg["training"]["num_epochs"],
        cvs=cfg["training"]["cvs"],
        num_epochs_refit=cfg["refit"]["num_epochs"],
        refit_iters=cfg["refit"]["iterations"],
        refit_resample=cfg["refit"]["resample"],
        optimizer_class=_resolve_torch_class(torch.optim, optimizer["name"], "optimizer"),
        lr=optimizer["learning_rate"],
        weight_decays=weight_decays,
        optimizerkw=optimizer.get("kwargs") or {},
        optimizer_paramskw=optimizer.get("parameter_kwargs") or {},
        scheduler_class=_resolve_torch_class(torch.optim.lr_scheduler, scheduler["name"], "scheduler"),
        scheduler_kwargs=scheduler.get("kwargs") or {},
        eps=eps,
        eps_factor=pruning["eps_factor"],
        fill_zeroed=pruning["fill_zeroed"],
        device=cfg["runtime"]["device"],
    )
