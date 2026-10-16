"""Configuration loading helpers."""
from pathlib import Path

import yaml


def load_section(path, section):
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    with config_path.open() as stream:
        config = yaml.safe_load(stream) or {}
    values = config.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"{section!r} must be a YAML mapping in {config_path}")
    return values
