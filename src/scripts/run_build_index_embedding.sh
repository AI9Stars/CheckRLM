#!/bin/bash

set -e

# Load the config file
source config.sh
export PROJECT_ROOT=$PROJECT_ROOT

RETRIEVAL_METHOD="embedding" 

echo "Building ${DATASET_NAME} index using ${RETRIEVAL_METHOD} method..."
CUDA_VISIBLE_DEVICES=0,1 python -u $PROJECT_ROOT/src/build_index/$RETRIEVAL_METHOD/index.py \
    --model $EMBEDDING_MODEL \
    --dataset $DATASET_NAME \
    --input_data $CORPUS_PATH \
    --chunk_size 512 \
    --chunk_overlap 0
echo "${DATASET_NAME} index built."