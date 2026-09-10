#!/usr/bin/env bash
set -euo pipefail

STEGF_PRE_SERVER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$STEGF_PRE_SERVER_SCRIPT_DIR/setup_preprocess.sh" --profile server "$@"
