#!/bin/bash

# Simple chat launcher script

MODEL_PATH="${1:-export/models/qwen3/model}"
EXECUTION_PROVIDER="${2:-cpu}"

echo "Starting chat with:"
echo "  Model: $MODEL_PATH"
echo "  Provider: $EXECUTION_PROVIDER"
echo ""

python simple_chat.py -m "$MODEL_PATH" -e "$EXECUTION_PROVIDER"
