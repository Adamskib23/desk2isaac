#!/usr/bin/env bash
set -euo pipefail

ROOT="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
exec bash "$ROOT/scripts/run_isaac_replay.sh" --mode dynamic --replay-speed 0.65 --hold-seconds 2 "$@"
