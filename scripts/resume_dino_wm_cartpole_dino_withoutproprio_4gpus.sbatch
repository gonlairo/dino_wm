#!/bin/bash
#SBATCH --job-name=tdmpc2-cartpole-resume
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/cartpole-resume-%j.out
#SBATCH --error=slurm/cartpole-resume-%j.err
#SBATCH --account=hai_1068

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR}/slurm"

# Run from repo root (submit from dino_wm repo root)
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT" || exit 1

# Existing run to resume from:
# /p/home/jusers/gonzalezlaiz1/juwels/project/repos/dino_wm/outputs/cartpole_dino_wm_job13268589/dino224x224_with_proprio
BASE_OUT="$(pwd)/outputs/cartpole_dino_wm_job13268585"


RUN_DIR="$BASE_OUT/dino224x224_without_proprio"
SLURM_LOGS="$BASE_OUT/slurm_resume_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR" "$SLURM_LOGS"

# Safety check: fail fast if no checkpoint is present.
if [ ! -f "$RUN_DIR/checkpoints/model_latest.pth" ]; then
  echo "Checkpoint not found: $RUN_DIR/checkpoints/model_latest.pth" >&2
  exit 1
fi

WANDB_MODE=offline
export WANDB_MODE
export WANDB_API_KEY=$(cat "/p/home/jusers/gonzalezlaiz1/juwels/project/repos/world-models-interp/.wandb_api_key")

source .venv/bin/activate
DATA_DIR_224x224="/p/home/jusers/gonzalezlaiz1/juwels/project/dino_wm_osf/osfstorage/datasets/data_off_policy_224x224_dino_wm"

# 224x224, DINO, with proprio, RESUME.
# NOTE: training.epochs is additional epochs after resume.
# Last completed epoch was 18, so 82 more epochs reaches total epoch 100.
accelerate launch --num_processes 4 train.py \
  --config-name train_local.yaml \
  env=cartpole \
  env.dataset.data_path="${DATA_DIR_224x224}" \
  env.dataset.use_proprio=false \
  frameskip=1 \
  num_hist=3 \
  num_pred=1 \
  training.epochs=82 \
  training.batch_size=32 \
  training.seed=1 \
  encoder=dino \
  hydra.run.dir="$RUN_DIR" \
  > "$SLURM_LOGS/dino224x224_with_proprio.out" 2> "$SLURM_LOGS/dino224x224_with_proprio.err" &

wait
