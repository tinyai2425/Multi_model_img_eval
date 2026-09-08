#!/bin/bash
# 一次生成 MMStar + OCRBench 的 GPU/API jsonl（合成各一份）。
# 同一 N 次抽样结果固定。N=0 表示跳过该数据集。
# enable_thinking 在本文件指定，写入用例（覆盖 model_config 里的同名字段）。

CONFIG_PATH="../model_config/Qwen3.5-4B-test-config.json"
MMSTAR_N=120
OCRBENCH_N=120
ENABLE_THINKING=false

# 默认读 ../Data_set/... ；若数据集在别处可在此覆盖
MMSTAR_PARQUET=""
OCRBENCH_PARQUET=""

CMD="python Gen_img_cases.py \"$CONFIG_PATH\" \"$MMSTAR_N\" \"$OCRBENCH_N\" --enable_thinking \"$ENABLE_THINKING\""
if [ -n "$MMSTAR_PARQUET" ]; then
    CMD="$CMD --mmstar_parquet \"$MMSTAR_PARQUET\""
fi
if [ -n "$OCRBENCH_PARQUET" ]; then
    CMD="$CMD --ocrbench_parquet \"$OCRBENCH_PARQUET\""
fi

eval $CMD
