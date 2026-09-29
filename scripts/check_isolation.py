"""Verify interpreter, environment, temporary files, and library cache locations."""
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
PATH_VARIABLES = (
    "UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR",
    "UV_PYTHON_CACHE_DIR", "UV_PYTHON_BIN_DIR", "UV_TOOL_DIR", "UV_TOOL_BIN_DIR",
    "UV_CREDENTIALS_DIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
    "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH", "HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_XET_CACHE",
    "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "PYTHONPYCACHEPREFIX", "PYTHONUSERBASE", "TMPDIR", "TMP", "TEMP",
)


def check():
    paths = {name: os.environ.get(name) for name in PATH_VARIABLES}
    paths.update(python=sys.executable, base_python=sys.base_prefix, virtualenv=sys.prefix,
                 torch_hub=torch.hub.get_dir(), temporary_files=tempfile.gettempdir())
    for name, value in paths.items():
        if not value or not Path(value).resolve().is_relative_to(ROOT):
            raise RuntimeError(f"Outside checkout or unset: {name}={value}. Use ./uv.sh with a fresh local .venv.")
    return {"status": "isolated", "root": str(ROOT), "paths": paths}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
