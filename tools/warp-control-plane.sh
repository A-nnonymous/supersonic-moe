#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEFAULT_WARP_ROOT=$(cd "$SCRIPT_DIR/../../warp" 2>/dev/null && pwd || true)
WARP_ROOT=${WARP_ROOT:-$DEFAULT_WARP_ROOT}

if [ -z "$WARP_ROOT" ] || [ ! -f "$WARP_ROOT/runtime/control_plane.py" ]; then
	echo "warp control plane not found. Set WARP_ROOT or clone https://github.com/A-nnonymous/warp.git at ../warp." >&2
	exit 1
fi

exec python3 "$WARP_ROOT/runtime/control_plane.py" "$@"