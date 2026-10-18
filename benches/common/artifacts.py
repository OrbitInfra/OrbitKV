"""Validate explicit experiment destinations before a harness writes evidence."""

from argparse import ArgumentTypeError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def external_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(ROOT):
        raise ArgumentTypeError(f"experiment output must be outside the source checkout: {path}")
    return path
