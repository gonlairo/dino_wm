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
mkdir -p "$BASE_OUT/resnet64x64_with_proprio" "$BASE_OUT/resnet64x64_without_proprio" "$BASE_OUT/dino224x224_with_proprio" "$BASE_OUT/dino224x224_without_proprio" "$SLURM_LOGS"

export WANDB_MODE=online
export WANDB_API_KEY=$(cat "/p/home/jusers/gonzalezlaiz1/juwels/project/repos/world-models-interp/.wandb_api_key")

source .venv/bin/activate

DATA_DIR_64x64="/p/home/jusers/gonzalezlaiz1/juwels/project/dino_wm_osf/osfstorage/datasets/data_off_policy_64x64_dino_wm"
DATA_DIR_224x224="/p/home/jusers/gonzalezlaiz1/juwels/project/dino_wm_osf/osfstorage/datasets/data_off_policy_224x224_dino_wm"

# RESNET --> (has_decoder=false: VQVAE outputs 16x16 from ResNet's 1-patch latent, not 64x64)

# 64x64, ResNet, use_proprio=true
srun --exclusive -n 1 --gres=gpu:1 \
  env WANDB_API_KEY="$WANDB_API_KEY" \
  WANDB_MODE=online \
  python train.py \
  --config-name train_local.yaml \
  env=cartpole \
  env.dataset.data_path="${DATA_DIR_64x64}" \
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


# 64x64, ResNet, use_proprio=false
srun --exclusive -n 1 --gres=gpu:1 \
  env WANDB_API_KEY="$WANDB_API_KEY" \
  WANDB_MODE=online \
  python train.py \
  --config-name train_local.yaml \
  env=cartpole \
  env.dataset.data_path="${DATA_DIR_64x64}" \
  env.dataset.use_proprio=false \
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
  hydra.run.dir="$BASE_OUT/resnet64x64_without_proprio" \
  > "$SLURM_LOGS/resnet64x64_without_proprio.out" 2> "$SLURM_LOGS/resnet64x64_without_proprio.err" &




# 224x224, DINO, use_proprio=true
srun --exclusive -n 1 --gres=gpu:1 \
  env WANDB_API_KEY="$WANDB_API_KEY" \
  WANDB_MODE=online \
  python train.py \
  --config-name train_local.yaml \
  env=cartpole \
  env.dataset.data_path="${DATA_DIR_224x224}" \
  frameskip=1 \
  num_hist=3 \
  num_pred=1 \
  training.epochs=100 \
  training.batch_size=32 \
  training.seed=1 \
  encoder=dino \
  hydra.run.dir="$BASE_OUT/dino224x224_with_proprio" \
  > "$SLURM_LOGS/dino224x224_with_proprio.out" 2> "$SLURM_LOGS/dino224x224_with_proprio.err" &


# 224x224, DINO, use_proprio=false
srun --exclusive -n 1 --gres=gpu:1 \
  env WANDB_API_KEY="$WANDB_API_KEY" \
  WANDB_MODE=online \
  python train.py \
  --config-name train_local.yaml \
  env=cartpole \
  env.dataset.data_path="${DATA_DIR_224x224}" \
  env.dataset.use_proprio=false \
  frameskip=1 \
  num_hist=3 \
  num_pred=1 \
  training.epochs=100 \
  training.batch_size=32 \
  training.seed=1 \
  encoder=dino \
  hydra.run.dir="$BASE_OUT/dino224x224_without_proprio" \
  > "$SLURM_LOGS/dino224x224_without_proprio.out" 2> "$SLURM_LOGS/dino224x224_without_proprio.err" &

wait
