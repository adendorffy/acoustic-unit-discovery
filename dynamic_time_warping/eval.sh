#!/bin/bash

MODEL_NAMES=("wavlm_base" "wavlm_large" "wavlm_xlarge" "hubert_base" "hubert_large" "hubert_xlarge")
LAYER_NUMBERS=(6 7 8 9)

ENCODINGS_DIR="encodings/librispeech_subset/"
ALIGNMENTS_DIR="data/all_alignments/"
OUTPUT_DIR="output/cython_dtw/"
EVAL_ALIGNMENTS="data/librispeech_subset_alignments/words_and_indices.txt"
RESULTS_FILE="output/cython_dtw/new_results.csv"

for MODEL in "${MODEL_NAMES[@]}"; do
    for LAYER in "${LAYER_NUMBERS[@]}"; do
        echo "Clustering $MODEL $LAYER"
        python cluster.py "$OUTPUT_DIR" "$OUTPUT_DIR/new_clusters/" "$MODEL" "$LAYER" 

        echo "Evaluating $MODEL $LAYER"
        if [[ "$MODEL" == "wavlm_base" && "$LAYER" == 6 ]]; then
            python eval.py "$EVAL_ALIGNMENTS" "$OUTPUT_DIR/new_clusters/" "$MODEL" "$LAYER" "$RESULTS_FILE" "--write_header"
        else
            python eval.py "$EVAL_ALIGNMENTS" "$OUTPUT_DIR/new_clusters/" "$MODEL" "$LAYER" "$RESULTS_FILE"
        fi
    done
done
