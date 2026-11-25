#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaac="${ISAAC_SIM_ROOT:-$HOME/Documents/isaacSim}"
script="$1"
shift
runtime="$(mktemp -d "${TMPDIR:-/tmp}/simforklift-kit.XXXXXXXX")"
trap 'rm -rf "$runtime"' EXIT
export SIMFORKLIFT_ROOT="$root"
export PYTHONDONTWRITEBYTECODE=1
export SIMFORKLIFT_ARGS="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@")"
cd "$root"
"$isaac/kit/kit" "$root/scripts/simulation.kit" --no-window \
  --ext-folder "$isaac/exts" --ext-folder "$isaac/extscache" --ext-folder "$isaac/extsDeprecated" \
  --portable --portable-root "$runtime" --exec "scripts/$script"
