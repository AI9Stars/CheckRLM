#!/bin/bash

set -e

# Load the config file
source config.sh
export PROJECT_ROOT=$PROJECT_ROOT
export ES_URL=$ES_URL

RETRIEVAL_METHOD="bm25" 

echo "Building ${DATASET_NAME} index using ${RETRIEVAL_METHOD} method..."
python -u $PROJECT_ROOT/src/build_index/$RETRIEVAL_METHOD/index.py \
    --index_name $DATASET_NAME \
    --input_data $CORPUS_PATH
echo "${DATASET_NAME} index built."