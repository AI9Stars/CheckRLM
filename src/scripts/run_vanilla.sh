#!/bin/bash

set -e

# Load the config file
source config.sh
export PROJECT_ROOT=$PROJECT_ROOT
export VLLM_API_KEY=$VLLM_API_KEY 
export REASONING_BASE_URL=$REASONING_BASE_URL
export ES_URL=$ES_URL

METHOD="vanilla"
RETRIEVAL_METHOD="embedding" # choices=["bm25", "embedding"]

# Import functions and scripts
source cleanup.sh

trap cleanup EXIT

# Start elasticsearch engine (source so ES_PID is visible to cleanup trap)
source "$(dirname "$0")/start_elasticsearch.sh"

# Start reasoning model server
source "$(dirname "$0")/start_reasoning_model.sh"

# Run the script
echo "Start running ${METHOD} script..."
echo "Project root: ${PROJECT_ROOT}, VLLM API Key: ${VLLM_API_KEY}, Reasoning base url: ${REASONING_BASE_URL}, Elasticsearch url: ${ES_URL}"

python -u $PROJECT_ROOT/src/tasks/base.py \
    --reasoning_model_path $REASONING_MODEL_PATH \
    --retrieval_method $RETRIEVAL_METHOD \
    --method $METHOD \
    --dataset_name $DATASET_NAME \
    --dataset_file $DATASET_FILE \
    --embedding_model $EMBEDDING_MODEL \
    --corpus_path $CORPUS_PATH \
    --vector_path $VECTOR_PATH \
    --TopK $TopK \
    --evaluate \
    --debug_mode

echo "${METHOD} script finished."

echo "Shutting down the reasoning model server..."
kill $REASONING_PID
echo "Reasoning model server shut down."

echo "Shutting down the elasticsearch engine..."
kill $ES_PID
echo "Elasticsearch engine shut down."

trap - EXIT