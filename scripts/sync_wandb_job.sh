#!/usr/bin/env bash
# Sync all wandb offline runs from a cartpole_dino_wm job output directory to wandb.ai.
# Usage:
#   ./scripts/sync_wandb_job.sh outputs/cartpole_dino_wm_job13268589
#   ./scripts/sync_wandb_job.sh /path/to/outputs/cartpole_dino_wm_job13268589
#
# Optional: set WANDB_API_KEY_FILE to override the default API key path.
#   WANDB_API_KEY_FILE=/path/to/.wandb_api_key ./scripts/sync_wandb_job.sh outputs/cartpole_dino_wm_job13268589

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_WANDB_API_KEY_FILE="${DEFAULT_WANDB_API_KEY_FILE:-/p/home/jusers/gonzalezlaiz1/juwels/project/repos/world-models-interp/.wandb_api_key}"
WANDB_API_KEY_FILE="${WANDB_API_KEY_FILE:-$DEFAULT_WANDB_API_KEY_FILE}"

usage() {
  echo "Usage: $0 <output_dir>" >&2
  echo "  output_dir  e.g. outputs/cartpole_dino_wm_job13268589 (relative to repo root or absolute)" >&2
  echo "" >&2
  echo "Optional env:" >&2
  echo "  WANDB_API_KEY_FILE  path to file containing WANDB_API_KEY (default: $DEFAULT_WANDB_API_KEY_FILE)" >&2
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

OUTPUT_DIR="$1"
if [[ ! "$OUTPUT_DIR" = /* ]]; then
  OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Error: output directory does not exist: $OUTPUT_DIR" >&2
  exit 1
fi

if [[ ! -f "$WANDB_API_KEY_FILE" ]]; then
  echo "Error: WANDB API key file not found: $WANDB_API_KEY_FILE" >&2
  exit 1
fi

export WANDB_API_KEY
WANDB_API_KEY=$(cat "$WANDB_API_KEY_FILE")

cd "$REPO_ROOT" || exit 1
if [[ -d ".venv" ]]; then
  source .venv/bin/activate
fi

# Find all wandb offline run dirs: .../wandb/offline-run-*
run_dirs=()
while IFS= read -r -d '' d; do
  run_dirs+=("$d")
done < <(find "$OUTPUT_DIR" -type d -name "offline-run-*" -path "*/wandb/offline-run-*" -print0 2>/dev/null | sort -z)

if [[ ${#run_dirs[@]} -eq 0 ]]; then
  echo "No wandb offline runs found under $OUTPUT_DIR"
  exit 0
fi

echo "Found ${#run_dirs[@]} wandb offline run(s). Syncing..."
for run_dir in "${run_dirs[@]}"; do
  echo "Syncing: $run_dir"
  wandb sync "$run_dir" || echo "Warning: sync failed for $run_dir"
done
echo "Done."
