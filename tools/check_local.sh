#!/usr/bin/env bash
# Everything CI checks, plus the GPU tests CI cannot run. Run before every commit:
#
#   source .venv/bin/activate
#   tools/check_local.sh
#
# Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n==> %s\n' "$*"; }

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "error: activate the venv first: source .venv/bin/activate" >&2
    exit 1
fi

step "Workflow files (actionlint)"
actionlint

step "Python lint and format (ruff)"
ruff check .
ruff format --check .

step "C++/CUDA build, warnings as errors"
cmake -S . -B build -DCMAKE_COMPILE_WARNING_AS_ERROR=ON > /dev/null
cmake --build build -j

step "Toolchain smoke test"
./build/spectral_smoke

step "PyTorch extension"
pip install -e . --no-build-isolation --quiet

step "Tests, including the GPU tests"
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'no GPU visible')"
pytest -q

printf '\nAll checks passed.\n'