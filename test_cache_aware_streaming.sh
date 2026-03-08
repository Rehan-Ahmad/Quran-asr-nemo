#!/usr/bin/env bash
set -euo pipefail

# Test cache-aware streaming inference on Quran dataset
# Uses the official NeMo streaming inference script

WORKSPACE_ROOT=$(pwd)
PYTHON_BIN="${WORKSPACE_ROOT}/.venv/bin/python"
SCRIPT_PATH="${WORKSPACE_ROOT}/nemo_scripts/speech_to_text_cache_aware_streaming_infer.py"

# Model paths
STREAMING_MODEL="${WORKSPACE_ROOT}/nemo_experiments/FastConformer-English-Quran-Tokenizer/finetune/2026-02-18_15-04-25/checkpoints/FastConformer-English-Quran-Tokenizer-streaming.nemo"
TEST_MANIFEST="${WORKSPACE_ROOT}/data/manifests/test.json"
OUTPUT_DIR="${WORKSPACE_ROOT}/cache_aware_streaming_results"

echo "========================================="
echo "CACHE-AWARE STREAMING INFERENCE TEST"
echo "========================================="
echo "Model: FastConformer-English-Quran-Tokenizer-streaming.nemo"
echo "Dataset: Quran Test Set"
echo "Script: speech_to_text_cache_aware_streaming_infer.py"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test 1: Single audio file (quick test)
echo ""
echo "Test 1: Single Audio File Test"
echo "-----------------------------------------"

# Create single-sample manifest in workspace root (not in output dir - avoids Hydra path issues)
SINGLE_MANIFEST="${WORKSPACE_ROOT}/test_single_cache.json"
head -1 "$TEST_MANIFEST" > "$SINGLE_MANIFEST"

FIRST_AUDIO=$(head -1 "$SINGLE_MANIFEST" | python3 -c "import sys, json; print(json.load(sys.stdin)['audio_filepath'])")
echo "Audio file: $FIRST_AUDIO"

"$PYTHON_BIN" "$SCRIPT_PATH" \
    model_path="$STREAMING_MODEL" \
    dataset_manifest="$SINGLE_MANIFEST" \
    batch_size=1 \
    compare_vs_offline=true \
    amp=false \
    debug_mode=true

echo ""
echo "✓ Single file test completed!"

# Test 2: Small batch from manifest (first 10 samples)
echo ""
echo "Test 2: Small Batch Test (10 samples)"
echo "-----------------------------------------"

# Create temporary manifest with first 10 samples in workspace root
TEMP_MANIFEST="${WORKSPACE_ROOT}/test_small_cache.json"
head -10 "$TEST_MANIFEST" > "$TEMP_MANIFEST"

"$PYTHON_BIN" "$SCRIPT_PATH" \
    model_path="$STREAMING_MODEL" \
    dataset_manifest="$TEMP_MANIFEST" \
    batch_size=4 \
    compare_vs_offline=true \
    amp=false \
    debug_mode=false \
    output_path="$OUTPUT_DIR"

echo ""
echo "✓ Small batch test completed!"

# Test 3: Different attention contexts
echo ""
echo "Test 3: Testing Different Attention Contexts"
echo "-----------------------------------------"

for ctx in "[70,13]" "[70,6]" "[70,1]" "[70,0]"; do
    echo ""
    echo "Testing with context: $ctx"
    
    "$PYTHON_BIN" "$SCRIPT_PATH" \
        model_path="$STREAMING_MODEL" \
        dataset_manifest="$SINGLE_MANIFEST" \
        batch_size=1 \
        att_context_size="$ctx" \
        compare_vs_offline=false \
        amp=false \
        debug_mode=false
done

echo ""
echo "✓ All context tests completed!"

# Optional: Full test set evaluation (comment out to skip)
echo ""
echo "Test 4: Full Test Set Evaluation"
echo "-----------------------------------------"
echo "This will take longer (3,745 samples)..."

# Create full manifest copy in workspace root so relative audio paths resolve correctly
FULL_MANIFEST="${WORKSPACE_ROOT}/test_full_cache.json"
cp "$TEST_MANIFEST" "$FULL_MANIFEST"

"$PYTHON_BIN" "$SCRIPT_PATH" \
    model_path="$STREAMING_MODEL" \
    dataset_manifest="$FULL_MANIFEST" \
    batch_size=32 \
    compare_vs_offline=false \
    amp=false \
    debug_mode=false \
    output_path="$OUTPUT_DIR"

echo ""
echo "✓ Full evaluation completed!"

echo ""
echo "========================================="
echo "✅ ALL TESTS COMPLETED SUCCESSFULLY!"
echo "========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To run full test set evaluation (3,745 samples):"
echo "  Uncomment Test 4 section in this script"
echo ""
echo "To test on your own audio file:"
echo "  # Create a single-sample manifest"
echo "  echo '{\"audio_filepath\": \"path/to/your/audio.wav\", \"duration\": 10.0, \"text\": \"reference text\"}' > my_test.json"
echo "  $PYTHON_BIN nemo_scripts/speech_to_text_cache_aware_streaming_infer.py \\"
echo "    model_path=$STREAMING_MODEL \\"
echo "    dataset_manifest=my_test.json \\"
echo "    batch_size=1 \\"
echo "    compare_vs_offline=true \\"
echo "    debug_mode=true"
echo ""

# Cleanup temporary manifests
rm -f "${WORKSPACE_ROOT}/test_single_cache.json" "${WORKSPACE_ROOT}/test_small_cache.json" "${WORKSPACE_ROOT}/test_full_cache.json"
