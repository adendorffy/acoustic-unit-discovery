#!/bin/bash

MODEL_NAMES=("wavlm_base" "wavlm_large" "wavlm_xlarge" "hubert_base" "hubert_large" "hubert_xlarge")
LAYER_NUMBERS=(6 7 8 9)
DISTANCE_THRESHOLDS=(0.5 0.55 0.6)

ENCODINGS_DIR="encodings/librispeech-wav/"
ALIGNMENTS_DIR="data/all_alignments/"
OUTPUT_DIR="full_output/cython_dtw/"
EVAL_ALIGNMENTS="data/words_and_indices.txt"
RESULTS_FILE="output/dtw/results.csv"

for MODEL in "${MODEL_NAMES[@]}"; do
    for LAYER in "${LAYER_NUMBERS[@]}"; do
        echo "Running DTW on $MODEL $LAYER"

        python dtw.py "$ENCODINGS_DIR" "$ALIGNMENTS_DIR" "$OUTPUT_DIR" "$MODEL" "$LAYER"
    
    done
done
