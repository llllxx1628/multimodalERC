#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-"erc_only"}
PYTHON=${PYTHON:-python}
CONFIG_FILE=${CONFIG_FILE:-"config.json"}
MODALITIES=${MODALITIES:-"v a t"}

echo "---------------------------------------------"
echo " Training pipeline for IEMOCAP"
echo " Mode        : ${MODE}"
echo " Config file : ${CONFIG_FILE}"
echo " Modalities  : ${MODALITIES}"
echo "---------------------------------------------"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config file not found: ${CONFIG_FILE}"
  exit 1
fi

if [ "${MODE}" = "from_scratch" ]; then

  echo "Step 1: Training LipSync network on IEMOCAP..."
  ${PYTHON} train_lipsync_iemocap.py

  echo "Step 2: Extracting text, audio, and visual features for IEMOCAP..."
  ${PYTHON} feature_extract_iemocap.py

  echo "Step 3: Training the multimodal ERC model on IEMOCAP..."
  ${PYTHON} train_erc_iemocap.py \
      --modalities ${MODALITIES} \
      --config "${CONFIG_FILE}"

elif [ "${MODE}" = "erc_only" ]; then
  echo "Running ERC training on IEMOCAP with pre-extracted features..."
  ${PYTHON} train_erc_iemocap.py \
      --dataset iemocap \
      --modalities ${MODALITIES} \
      --config "${CONFIG_FILE}"

else
  echo "Unknown mode: ${MODE}"
  echo "Usage:"
  echo "  bash train_iemocap.sh from_scratch"
  echo "  bash train_iemocap.sh erc_only"
  exit 1
fi

echo "Done."
