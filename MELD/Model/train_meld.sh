set -euo pipefail

MODE=${1:-"erc_only"}
PYTHON=${PYTHON:-python}
CONFIG_FILE=${CONFIG_FILE:-"ERC/MELD/Dataset/config.json"}
MODALITIES=${MODALITIES:-"v a t"}

echo "---------------------------------------------"
echo " Training pipeline for MELD"
echo " Mode        : ${MODE}"
echo " Config file : ${CONFIG_FILE}"
echo " Modalities  : ${MODALITIES}"
echo "---------------------------------------------"

if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Config file not found: ${CONFIG_FILE}"
  exit 1
fi

if [ "${MODE}" = "from_scratch" ]; then
  echo "Step 1: Extracting face videos and audio..."
  ${PYTHON} video_audio_extract.py

  echo "Step 2: Training LipSync network..."
  ${PYTHON} train_lipsync.py

  echo "Step 3: Extracting text, audio, and visual features..."
  ${PYTHON} feature_extract.py

  echo "Step 4: Training the multimodal ERC model..."
  ${PYTHON} train_erc.py \
      --modalities ${MODALITIES} \
      --config "${CONFIG_FILE}"

elif [ "${MODE}" = "erc_only" ]; then
  echo "Running ERC training with pre-extracted features..."
  ${PYTHON} train_erc.py \
      --modalities ${MODALITIES} \
      --config "${CONFIG_FILE}"

else
  echo "Unknown mode: ${MODE}"
  echo "Usage:"
  echo "  bash train.sh from_scratch"
  echo "  bash train.sh erc_only"
  exit 1
fi

echo "Done."
