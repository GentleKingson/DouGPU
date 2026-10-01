#!/usr/bin/env bash
# Install DouGPU only into this project's .venv. No global Python edits.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode=gpu
python_bin="${DOUGPU_PYTHON:-python3.12}"

usage() {
  cat <<'EOF'
Usage: bash scripts/install.sh [--gpu|--cpu] [--python /path/to/python3.12]

Creates or reuses DouGPU's project-local Python 3.12 .venv.
  --gpu     Install the pinned JAX + CUDA 13.0 wheel stack (default).
  --cpu     Install CPU JAX for correctness checks; no CUDA packages.
  --python  Select an existing Python 3.12 executable.

Supported training hosts: native Linux x86_64, or Ubuntu under Windows WSL2.
This installer does not install Python, system CUDA, or an NVIDIA driver.
EOF
}

while (($#)); do
  case "$1" in
    --gpu) mode=gpu; shift ;;
    --cpu) mode=cpu; shift ;;
    --python)
      if (($# < 2)); then
        printf '%s\n' 'Error: --python requires a Python 3.12 executable.' >&2
        exit 2
      fi
      python_bin="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != Linux ]]; then
  printf '%s\n' 'Use Linux or Ubuntu under WSL2. Native Windows CUDA JAX is unsupported.' >&2
  exit 2
fi
if [[ "$(uname -m)" != x86_64 ]]; then
  printf '%s\n' 'This RTX 5070 migration package targets Linux x86_64.' >&2
  exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Python executable not found: %s. Use --python with an existing Python 3.12.\n' "$python_bin" >&2
  exit 2
fi
"$python_bin" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Python 3.12 is required by this pinned environment.")'

venv_dir="$project_dir/.venv"
if [[ -L "$venv_dir" ]]; then
  printf '%s\n' 'Error: .venv is a symbolic link; use a virtual environment within this project.' >&2
  exit 2
fi
if [[ -e "$venv_dir" && ! -f "$venv_dir/pyvenv.cfg" ]]; then
  printf '%s\n' 'Error: .venv exists but is not a Python virtual environment; left unchanged.' >&2
  exit 2
fi
if [[ ! -f "$venv_dir/pyvenv.cfg" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi
venv_python="$venv_dir/bin/python"
"$venv_python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Existing .venv is not Python 3.12; use a new project directory or replace that environment yourself.")'

# Avoid silently turning a pre-existing GPU environment into a claimed CPU-only
# environment. Installation of missing GPU packages into a CPU .venv is allowed.
if [[ "$mode" == cpu ]]; then
  "$venv_python" - <<'PY'
import importlib.metadata as md
names = {d.metadata.get("Name", "").lower().replace("_", "-") for d in md.distributions()}
if names & {"jax-cuda12-plugin", "jax-cuda13-plugin"}:
    raise SystemExit("This .venv already has a CUDA JAX plugin. Use a new project directory for a CPU-only install.")
PY
fi

if [[ "$mode" == gpu ]]; then
  "$venv_python" - <<'PY'
import importlib.metadata as md
names = {d.metadata.get("Name", "").lower().replace("_", "-") for d in md.distributions()}
if "jax-cuda12-plugin" in names:
    raise SystemExit("This .venv has the CUDA 12 JAX plugin. Use a new project directory for the pinned CUDA 13 install.")
PY
fi

"$venv_python" -m pip install --disable-pip-version-check --only-binary=:all: \
  -r "$project_dir/requirements-$mode.txt"
"$venv_python" -m pip check

# Record what was actually installed on this host, including installation tools.
# This is generated on the user's machine, not a substituted validation record.
freeze_path="$venv_dir/resolved-$mode.txt"
"$venv_python" -m pip freeze --all > "$freeze_path"
printf '\nInstalled into %s\nResolved versions: %s\n' "$venv_dir" "$freeze_path"
printf 'Activate: source "%s/bin/activate"\n' "$venv_dir"
if [[ "$mode" == gpu ]]; then
  printf '%s\n' 'Next: python run_local.py doctor' 'Then: python run_local.py --help'
else
  printf '%s\n' 'Next: python run_local.py --help (use the CPU smoke profile for checks)'
fi
