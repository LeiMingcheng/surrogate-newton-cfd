#!/usr/bin/env python3
"""AeroOpt calculation-directory entrypoint for the unified framework."""

from __future__ import annotations

import os
from pathlib import Path

from optimization.config import load_optimization_config
from optimization.objective import evaluate_workdir


def main() -> int:
    config_path = os.environ.get("SURROGATE_NEWTON_OPT_CONFIG")
    if not config_path:
        config_path = str((Path.cwd() / ".." / ".." / "optimization_config.json").resolve())
    evaluate_workdir(Path.cwd(), load_optimization_config(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
