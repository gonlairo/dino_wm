#!/usr/bin/env bash
# Download DINO-WM datasets from OSF into your project directory.
# OSF project: https://osf.io/bmw48/ (view_only link in README)
set -euo pipefail

PROJECT_DIR="${1:-/p/home/jusers/gonzalezlaiz1/juwels/project}"
OSF_PROJECT_ID="bmw48"
DEST="${PROJECT_DIR}/dino_wm_osf"

echo "Destination: ${DEST}"
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

# Install osfclient if not available
if ! command -v osf &>/dev/null; then
  echo "Installing osfclient..."
  pip install osfclient
fi

# Clone the OSF project (for view-only you may need -u YOUR_OSF_EMAIL and password when prompted)
echo "Cloning OSF project ${OSF_PROJECT_ID}..."
osf -p "${OSF_PROJECT_ID}" clone "${DEST}" || {
  echo "If this failed due to access: use an OSF account with access and run:"
  echo "  osf -p ${OSF_PROJECT_ID} -u YOUR_OSF_EMAIL clone ${DEST}"
  exit 1
}

# Unzip datasets as per README (deformable is multi-part)
echo "Unzipping archives..."
cd "${DEST}"
for f in *.zip; do
  [ -f "$f" ] || continue
  if [[ "$f" == deformable*zip ]]; then
    echo "Combining and unzipping deformable (multi-part)..."
    zip -s- "$f" -O deformable_full.zip && unzip -o deformable_full.zip && rm -f deformable_full.zip
  else
    unzip -o "$f"
  fi
done

echo "Done. Set DATASET_DIR to the folder that contains point_maze, pusht_noise, wall_single, deformable, etc."
echo "  export DATASET_DIR=${DEST}"
echo "  # or, if the structure is ${DEST}/data:"
echo "  export DATASET_DIR=${DEST}/data"
