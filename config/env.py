"""Portable environment-variable configuration shared by task modules."""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]


def _value(var, default=""):
    return os.environ.get(var, default)


def _path(var, default=""):
    value = _value(var, default)
    return Path(value).expanduser() if value else Path(f"/set-{var}")


def require_path(var):
    """Return a configured path or stop with an actionable message."""
    value = os.environ.get(var)
    if not value:
        raise SystemExit(f"{var} is required; set it as shown in .env.example")
    return Path(value).expanduser()


PROJECT = _path("EIB_ROOT", str(_HERE))
PYTHON = _value("EIB_PYTHON", sys.executable)
PYTHON_GPU = _value("EIB_PYTHON_GPU", "python")
AF3_MODEL_DIR = _path("EIB_AF3_MODEL_DIR")
AF3_DB_DIR = _path("EIB_AF3_DB_DIR")
ESM_PREFIX = _path("EIB_ESM_PREFIX")
AF3_MODULE = _value("EIB_AF3_MODULE", "alphafold/3.0.2")
HF_HOME = _path("EIB_HF_HOME")
BEPIPRED_CLI = _path("EIB_BEPIPRED_CLI")
UPSTREAM = _path("EIB_UPSTREAM")
UPSTREAM_FOLDS = _path("EIB_UPSTREAM_FOLDS")
MAIL_USER = _value("EIB_MAIL_USER")
