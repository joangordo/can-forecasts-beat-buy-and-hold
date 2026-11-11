"""Load config.yaml and apply optional overrides."""
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(base, extra):
    out = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path=None, quick=False, overrides=None):
    """Return the configuration as a nested dict.

    quick=True is a fast end-to-end check: quarterly instead of monthly refits,
    fewer bootstrap draws and only the last two robustness windows. Its output
    goes to a separate folder and is never published."""
    with open(Path(path) if path else ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if quick:
        cfg = _merge(cfg, {"models": {"refit_every": 63}, "stats": {"bootstrap_reps": 300}})
        cfg["robustness"]["windows"] = cfg["robustness"]["windows"][-2:]
    cfg["quick"] = bool(quick)
    return _merge(cfg, overrides or {})
