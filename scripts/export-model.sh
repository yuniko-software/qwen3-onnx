#!/bin/bash

# Model export script using Olive auto-optimization for LLM and Optimum for embeddings

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

show_help() {
    cat << EOF
Usage: ./export-model.sh [OPTIONS]

Export Qwen3 models (LLM or embedding) to ONNX format with optimization.

OPTIONS:
    -t, --type TYPE          Model type: llm or embedding (default: both)
    -p, --precision PREC     Precision: int4, int8, fp16, fp32 (default: int4)
    -d, --device DEVICE      Device: cpu, cuda (default: cpu)
    -o, --output DIR         Output directory (default: ../models/)
    -f, --force              Force re-export even if model already exists
    -h, --help               Show this help message

EXAMPLES:
    # Export both LLM and embedding models with defaults (int4, cpu)
    ./export-model.sh

    # Export only LLM model with fp16 precision
    ./export-model.sh --type llm --precision fp16

    # Export only embedding model
    ./export-model.sh --type embedding

    # Export both with CUDA support
    ./export-model.sh --device cuda --precision fp16

DEFAULT MODELS:
    LLM:       Qwen/Qwen3-0.6B
    Embedding: Qwen/Qwen3-Embedding-0.6B
EOF
}

# Default values
MODEL_TYPE="both"
PRECISION="int4"
DEVICE="cpu"
OUTPUT_DIR="$SCRIPT_DIR/../models"
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -p|--precision)
            PRECISION="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set execution provider based on device
if [ "$DEVICE" = "cuda" ]; then
    PROVIDER="CUDAExecutionProvider"
else
    PROVIDER="CPUExecutionProvider"
fi

# Export LLM using Olive with model builder
export_llm() {
    local model_name=$1
    local output_path=$2

    if [ "$FORCE" = false ] && [ -d "$output_path/model" ]; then
        echo "=========================================="
        echo "LLM model already exists at $output_path/model"
        echo "Skipping export (use --force to re-export)"
        echo "=========================================="
        echo ""
        return
    fi

    echo "=========================================="
    echo "Exporting LLM model"
    echo "  Model: $model_name"
    echo "  Output: $output_path"
    echo "  Device: $DEVICE"
    echo "  Provider: $PROVIDER"
    echo "  Precision: $PRECISION"
    echo "=========================================="
    echo ""

    olive auto-opt \
        --model_name_or_path "$model_name" \
        --output_path "$output_path" \
        --device "$DEVICE" \
        --provider "$PROVIDER" \
        --precision "$PRECISION" \
        --use_model_builder \
        --log_level 1

    echo ""
    echo "LLM model exported successfully to $output_path"
    echo ""
}

# Export embedding model using Optimum
export_embedding() {
    local model_name=$1
    local output_path=$2

    if [ "$FORCE" = false ] && [ -d "$output_path/model" ]; then
        echo "=========================================="
        echo "Embedding model already exists at $output_path/model"
        echo "Skipping export (use --force to re-export)"
        echo "=========================================="
        echo ""
        return
    fi

    echo "=========================================="
    echo "Exporting Embedding model"
    echo "  Model: $model_name"
    echo "  Output: $output_path/model"
    echo "  Using: optimum-cli (encoder-only model)"
    echo "=========================================="
    echo ""

    # Create output directory if it doesn't exist
    mkdir -p "$output_path/model"

    # Export using optimum-cli for encoder-only models
    optimum-cli export onnx \
        --model "$model_name" \
        --task feature-extraction \
        "$output_path/model"

    echo ""
    echo "Embedding model exported successfully to $output_path/model"
    echo ""
}

# Export models based on type
if [ "$MODEL_TYPE" = "llm" ] || [ "$MODEL_TYPE" = "both" ]; then
    LLM_MODEL="Qwen/Qwen3-0.6B"
    LLM_OUTPUT="$OUTPUT_DIR/qwen3-llm"
    export_llm "$LLM_MODEL" "$LLM_OUTPUT"
fi

if [ "$MODEL_TYPE" = "embedding" ] || [ "$MODEL_TYPE" = "both" ]; then
    EMBED_MODEL="Qwen/Qwen3-Embedding-0.6B"
    EMBED_OUTPUT="$OUTPUT_DIR/qwen3-embedding"
    export_embedding "$EMBED_MODEL" "$EMBED_OUTPUT"
fi

echo "=========================================="
echo "All exports completed successfully!"
echo "=========================================="
