#!/usr/bin/env bash
set -euo pipefail

ROOT="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}"
if [[ ! -x "$ISAAC_PYTHON" ]]; then
    printf 'Set ISAAC_PYTHON to your Isaac Lab Python executable.\n' >&2
    exit 1
fi

export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export __VK_LAYER_NV_optimus="${__VK_LAYER_NV_optimus:-NVIDIA_only}"

exec "$ISAAC_PYTHON" "$ROOT/scripts/replay_in_isaac_lab.py" \
    --trajectory "$ROOT/outputs/box_pick_place_001/stabilized/trajectory.json" \
    --device cuda:0 --hold-seconds 8 "$@"
