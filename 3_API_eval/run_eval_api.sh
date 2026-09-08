#!/bin/bash

# 解析鸿蒙 PC 上 run_chat.sh 产出的日志（一份 txt，内含 MMStar + OCRBench）
TEST_RESULT_PATH="../../model-eval-storage/Qwen3.5-4B/project-1/project-1-IMG-API-120-120.txt"
SHOW_DETAIL=true

if [ "$SHOW_DETAIL" = true ]; then
    python Eval_API_results.py "$TEST_RESULT_PATH" --show_detail
else
    python Eval_API_results.py "$TEST_RESULT_PATH"
fi
