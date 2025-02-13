#!/bin/bash

MODEL_NAMES=("wavlm_base" "wavlm_large" "wavlm_xlarge" "hubert_base" "hubert_large" "hubert_xlarge")
LAYER_NUMBERS=(6 7 8 9)
DISTANCE_THRESHOLDS=(0.5 0.55 0.6)

ENCODINGS_DIR="encodings/librispeech_subset/"
ALIGNMENTS_DIR="data/all_alignments/"
OUTPUT_DIR="output/dtw/"
EVAL_ALIGNMENTS="data/librispeech_subset_alignments/words_and_indices.txt"
RESULTS_FILE="output/dtw/results.csv"

for MODEL in "${MODEL_NAMES[@]}"; do
    for LAYER in "${LAYER_NUMBERS[@]}"; do
        echo "Running DTW on $MODEL $LAYER"
        python dtw.py "$ENCODINGS_DIR" "$ALIGNMENTS_DIR" "$OUTPUT_DIR" "$MODEL" "$LAYER"

        for DIST in "${DISTANCE_THRESHOLDS[@]}"; do
            echo "Clustering with $DIST"
            python cluster.py "$OUTPUT_DIR" "$OUTPUT_DIR/clusters/" "$MODEL" "$LAYER" "$DIST"

            echo "Eval with $DIST"
            python eval.py "$EVAL_ALIGNMENTS" "$OUTPUT_DIR/clusters" "$MODEL" "$LAYER" "$DIST" "$RESULTS_FILE"
        done
    done
done
