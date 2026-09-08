#!/bin/bash

# 一次跑完 jsonl 里的 MMStar + OCRBench
INPUT_JSONL="../../model-eval-storage/Qwen3.5-4B/project-1/project-1-IMG-GPU-120-120.jsonl"
VLLM_IP="127.0.0.1"
VLLM_PORT=8895
VLLM_MODEL_ID="qwen35_4b"
VERSION_FLAG=1
SHOW_DETAIL="true"

CMD="python Eval_GPU_results.py \"$INPUT_JSONL\" \"$VLLM_IP\" \"$VLLM_PORT\" \"$VLLM_MODEL_ID\" \"$VERSION_FLAG\""
if [ "$SHOW_DETAIL" = "true" ]; then
    CMD="$CMD --show_detail"
fi

eval $CMD
