#!/bin/bash
# Every setting can be overridden from the environment, e.g.
#   IMAGE_FILE_PATH=my_photo.jpg PROMPT="caption en" DTYPE=bfloat16 ./launch_inference.sh

MODEL_PATH="${MODEL_PATH:-$HOME/projects/paligemma-weights/paligemma-3b-pt-224}"
PROMPT="${PROMPT:-this building is }"
IMAGE_FILE_PATH="${IMAGE_FILE_PATH:-test_images/pic1.jpeg}"
MAX_TOKENS_TO_GENERATE="${MAX_TOKENS_TO_GENERATE:-100}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
DO_SAMPLE="${DO_SAMPLE:-False}"
ONLY_CPU="${ONLY_CPU:-False}"
DTYPE="${DTYPE:-float32}"  # float32 (~12 GB), bfloat16 or float16 (~6 GB)

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "No PaliGemma checkpoint at $MODEL_PATH (set MODEL_PATH, see README)" >&2
    exit 1
fi
if [ ! -f "$IMAGE_FILE_PATH" ]; then
    echo "Image not found: $IMAGE_FILE_PATH (set IMAGE_FILE_PATH)" >&2
    exit 1
fi

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate "$MAX_TOKENS_TO_GENERATE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --do_sample "$DO_SAMPLE" \
    --only_cpu "$ONLY_CPU" \
    --dtype "$DTYPE"
