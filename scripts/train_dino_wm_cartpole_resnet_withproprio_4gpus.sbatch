#!/bin/bash
#SBATCH --job-name=tdmpc2-cartpole
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=slurm/cartpole-%j.out
#SBATCH --error=slurm/cartpole-%j.err
#SBATCH --account=hai_1068

set -euo pipefail

mkdir -p "${SLURM_SUBMIT_DIR}/slurm"

# Run from repo root (submit from dino_wm repo root)
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT" || exit 1

BASE_OUT="$(pwd)/outputs/cartpole_dino_wm_job${SLURM_JOB_ID}"
SLURM_LOGS="$BASE_OUT/slurm"
mkdir -p "$BASE_OUT/dino224x224_with_proprio" "$SLURM_LOGS"

WANDB_MODE=offline
export WANDB_MODE
export WANDB_API_KEY=$(cat "/p/home/jusers/gonzalezlaiz1/juwels/project/repos/world-models-interp/.wandb_api_key")

source .venv/bin/activate
DATA_DIR_64x64="/p/home/jusers/gonzalezlaiz1/juwels/project/dino_wm_osf/osfstorage/datasets/data_off_policy_64x64_dino_wm"


# 64x64, ResNet, with proprio
accelerate launch --num_processes 4 train.py \
--config-name train_local.yaml \
  env=cartpole \
  env.dataset.data_path="${DATA_DIR_64x64}" \
  env.dataset.use_proprio=true \
  img_size=64 \
  frameskip=1 \
  num_hist=3 \
  num_pred=1 \
  has_decoder=false \
  training.epochs=100 \
  training.batch_size=32 \
  training.seed=1 \
  encoder=resnet \
  encoder.pretrained=false \
  hydra.run.dir="$BASE_OUT/resnet64x64_with_proprio" \
  > "$SLURM_LOGS/resnet64x64_with_proprio.out" 2> "$SLURM_LOGS/resnet64x64_with_proprio.err" &

wait
