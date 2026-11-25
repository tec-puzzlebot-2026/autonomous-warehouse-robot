#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'TEXT'
Uso: bash run.sh [floor|first|upper] [--start] [--gui]

floor       Recogida de piso. Es la escena predeterminada.
first       Recogida desde la repisa de 20 mm.
upper       Escena de 200 mm con control ROS del elevador.

--start     Inicia la recogida al arrancar (floor y first).
            Por defecto espera el comando ROS Ready Pick Up.
--gui       Abre Isaac Sim con ventana. Por defecto ejecuta en CPU.
TEXT
}

scene=floor
if [[ ${1:-} != -* && $# -gt 0 ]]; then
  scene="$1"
  shift
fi
start=false
gui=false
for option in "$@"; do
  case "$option" in
    --start) start=true ;;
    --gui) gui=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Opción desconocida: %s\n' "$option" >&2; usage >&2; exit 2 ;;
  esac
done
case "$scene" in
  floor|first|upper) ;;
  *) printf 'Escena desconocida: %s\n' "$scene" >&2; usage >&2; exit 2 ;;
esac
if "$start" && [[ "$scene" != floor && "$scene" != first ]]; then
  printf '%s\n' '--start se usa con floor o first.' >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
isaac="${ISAAC_SIM_ROOT:-$HOME/Documents/isaacSim}"
bridge="$isaac/exts/isaacsim.ros2.bridge/jazzy"
if [[ ! -x "$isaac/kit/kit" || ! -d "$bridge/rclpy" ]]; then
  printf 'Se requiere Isaac Sim 5.0 con el puente Jazzy. Ajusta ISAAC_SIM_ROOT (actual: %s).\n' "$isaac" >&2
  exit 1
fi
export ISAAC_SIM_ROOT="$isaac" ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-74}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${TMPDIR:-/tmp}/puzzlebot-ros-$(id -u)}"
export LD_LIBRARY_PATH="$bridge/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SIMFORKLIFT_ROS_PYTHON="$bridge/rclpy"
export PYTHONPATH="$bridge/rclpy:$root/scripts"
export PYTHONDONTWRITEBYTECODE=1

args=(--world "$root/worlds/$scene.usda")
session="$(mktemp -d "${TMPDIR:-/tmp}/puzzlebot.XXXXXXXX")"
trap 'rm -rf -- "$session"' EXIT
args+=(--firmware dual --firmware-local --run-dir "$session")
if [[ "$scene" != upper ]]; then
  args+=(--pick-config "$root/config/pick.json" --pick-level "$scene" --execute-pick)
  if ! "$start"; then
    args+=(--wait-for-ready)
  fi
fi
if "$gui"; then
  "$isaac/python.sh" "$root/scripts/simulation.py" "${args[@]}"
else
  args+=(--cpu-rgb)
  bash "$root/scripts/run_kit.sh" cpu.py "${args[@]}"
fi
