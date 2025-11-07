#!/bin/bash

# Model export script using Olive auto-optimization
# Based on: https://onnxruntime.ai/docs/genai/tutorials/deepseek-python.html

MODEL_NAME="${1:-Qwen/Qwen3-0.6B}"                 # HuggingFace model name
OUTPUT_PATH="${2:-models/qwen3}"                # Output directory
DEVICE="${3:-cpu}"                              # Device: cpu, cuda
PROVIDER="${4:-CPUExecutionProvider}"           # Execution provider
PRECISION="${5:-int4}"                          # Precision: int4, int8, fp16, fp32

echo "Exporting model with following parameters:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_PATH"
echo "  Device: $DEVICE"
echo "  Provider: $PROVIDER"
echo "  Precision: $PRECISION"
echo ""

olive auto-opt \
  --model_name_or_path "$MODEL_NAME" \
  --output_path "$OUTPUT_PATH" \
  --device "$DEVICE" \
  --provider "$PROVIDER" \
  --precision "$PRECISION" \
  --use_model_builder \
  --log_level 1
