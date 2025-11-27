#!/bin/bash

# .NET test runner script

set -e

show_help() {
    cat << EOF
Usage: ./test-dotnet.sh [OPTIONS]

Run .NET tests for Qwen3 ONNX projects.

OPTIONS:
    -h, --help               Show this help message

EXAMPLES:
    # Run all tests
    ./test-dotnet.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOTNET_DIR="$SCRIPT_DIR/../dotnet"
MODELS_DIR="$SCRIPT_DIR/../models"

check_models() {
    local llm_model="$MODELS_DIR/qwen3-llm"
    local embedding_model="$MODELS_DIR/qwen3-embedding/model"
    local missing_models=false

    if [ ! -d "$llm_model" ]; then
        echo "Warning: LLM model not found at $llm_model"
        missing_models=true
    fi

    if [ ! -d "$embedding_model" ]; then
        echo "Warning: Embedding model not found at $embedding_model"
        missing_models=true
    fi

    if [ "$missing_models" = true ]; then
        echo ""
        echo "Models are missing. Running export-model.sh to generate them..."
        echo ""
        "$(dirname "$0")/export-model.sh"
        echo ""
    fi
}

run_tests() {
    local project_name=$1
    local project_path=$2

    echo "=========================================="
    echo "Running $project_name tests"
    echo "  Project: $project_path"
    echo "=========================================="
    echo ""

    dotnet test "$project_path"

    echo ""
    echo "$project_name tests completed"
    echo ""
}

echo "=========================================="
echo "Checking for required models"
echo "=========================================="
check_models

echo "=========================================="
echo "Running dotnet format verification"
echo "=========================================="
echo ""
dotnet format "$DOTNET_DIR/Qwen3.Onnx.Samples.slnx" --verify-no-changes
echo ""

run_tests "Embedding" "$DOTNET_DIR/Qwen3.Onnx.Embedding.Tests/Qwen3.Onnx.Embedding.Tests.csproj"
run_tests "LLM" "$DOTNET_DIR/Qwen3.Onnx.Llm.Tests/Qwen3.Onnx.Llm.Tests.csproj"

echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
