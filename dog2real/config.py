"""YAML config loading with ``extends:`` inheritance and deep merging."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: ``override`` merged recursively into ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(path: str | os.PathLike, relative_to: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    candidates = []
    if relative_to is not None:
        candidates.append(relative_to / p)
    candidates += [Path.cwd() / p, REPO_ROOT / p, CONFIG_DIR / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"config file not found: {path}")


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML file, following an optional top-level ``extends:`` key."""
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(load_yaml(resolve_path(parent, path.parent)), cfg)
    return cfg


def load_train_config(path: str | os.PathLike, overrides: dict | None = None) -> dict[str, Any]:
    """Load a training config and inline the referenced robot description."""
    cfg = load_yaml(path)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    robot = cfg.get("robot", "robot.yaml")
    if isinstance(robot, str):
        cfg["robot"] = load_yaml(robot)
    return cfg
