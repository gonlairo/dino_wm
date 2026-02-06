#!/usr/bin/env bash
set -euo pipefail

# Activate conda environment
source /home/hgf_hmgu/hgf_gib4562/miniconda3/etc/profile.d/conda.sh
conda activate dino_wm_minimal

# Dataset location
export DATASET_DIR="/home/hgf_hmgu/hgf_gib4562/dino_wm/data"
#mkdir -p "${DATASET_DIR}/cartpole"

# Create a random dataset: T=25,000, episode_len=500
python - <<'PY'
import torch
from pathlib import Path

out_dir = Path("/home/hgf_hmgu/hgf_gib4562/dino_wm/data/cartpole")
out_dir.mkdir(parents=True, exist_ok=True)

T = 1_000
episode_len = 500
assert T % episode_len == 0

images = torch.randint(0, 256, (T, 3, 64, 64), dtype=torch.uint8)
proprio = torch.randn(T, 4, dtype=torch.float32)
actions = torch.randn(T, 1, dtype=torch.float32)
states = proprio.clone()

payload = {
    "images": images,
    "proprio": proprio,
    "actions": actions,
    "states": states,
}

out_path = out_dir / "episodes.pth"
torch.save(payload, out_path)
print(f"Saved random dataset to {out_path}")
PY

# Train a quick model on the random dataset
python train.py \
  --config-name train_local.yaml \
  env=cartpole \
  img_size=64 \
  frameskip=1 \
  num_hist=2 \
  num_pred=1 \
  has_decoder=false \
  training.epochs=1 \
  training.batch_size=4 \
  encoder=resnet \
  encoder.pretrained=false
