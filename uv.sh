#!/usr/bin/env bash
# Execute uv with all project-owned runtime/cache state under this checkout.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Execute ./uv.sh; do not source it into your shell." >&2
  return 1
fi
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$project_root"
export HKD_PROJECT_ROOT="$project_root"
export UV_PROJECT="$project_root"
export UV_PROJECT_ENVIRONMENT="$project_root/.venv"
export UV_CACHE_DIR="$project_root/.local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$project_root/.local/python"
export UV_PYTHON_CACHE_DIR="$project_root/.local/python-cache"
export UV_PYTHON_BIN_DIR="$project_root/.local/bin"
export UV_TOOL_DIR="$project_root/.local/tools"
export UV_TOOL_BIN_DIR="$project_root/.local/bin"
export UV_CREDENTIALS_DIR="$project_root/.local/uv-credentials"
export UV_PYTHON="$(cat "$project_root/.python-version")"
export UV_MANAGED_PYTHON=1
export UV_PYTHON_INSTALL_BIN=0
export UV_PYTHON_INSTALL_REGISTRY=0
export UV_NO_MODIFY_PATH=1
export UV_LINK_MODE=copy
export XDG_CACHE_HOME="$project_root/.local/cache"
export XDG_CONFIG_HOME="$project_root/.local/config"
export XDG_DATA_HOME="$project_root/.local/share"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torch-inductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export PYTHONUSERBASE="$project_root/.local/python-user"
export PYTHONNOUSERSITE=1
export TMPDIR="$project_root/.local/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
# Ignore an activated environment without changing the parent shell.
unset VIRTUAL_ENV CONDA_PREFIX PYTHONHOME PYTHONPATH UV_NO_MANAGED_PYTHON UV_SYSTEM_PYTHON
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
exec uv "$@"
