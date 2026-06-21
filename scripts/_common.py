"""Shared helpers for the stage scripts."""

from __future__ import annotations

import argparse
import os

from publaynet_mmrag.config import Config, load_config

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "configs")


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Adds the standard config-selection arguments to a parser.

    Args:
        parser: The argument parser to extend.
    """
    parser.add_argument(
        "--config",
        default=os.path.join(_CONFIG_DIR, "base.yaml"),
        help="Path to the base YAML configuration.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "enhanced"],
        default="enhanced",
        help="Variant overlaid on the base config.",
    )


def resolve_config(args: argparse.Namespace) -> Config:
    """Loads the configuration selected by the parsed arguments.

    Args:
        args: Parsed arguments containing ``config`` and ``mode``.

    Returns:
        The validated :class:`~publaynet_mmrag.config.Config`.
    """
    variant = os.path.join(_CONFIG_DIR, f"{args.mode}.yaml")
    config = load_config(args.config, variant if os.path.exists(variant) else None)
    config.mode = args.mode
    return config
