#!/bin/bash

set -e

# Load the config file
source config.sh
export PROJECT_ROOT=$PROJECT_ROOT
export VLLM_API_KEY=$VLLM_API_KEY 
export REASONING_BASE_URL=$REASONING_BASE_URL
export CHECK_BASE_URL=$CHECK_BASE_URL

RETRIEVAL_METHOD="bm25" # choices = ["embedding", "bm25"]

# Import functions and scripts
source cleanup.sh

trap cleanup EXIT

# Start elasticsearch engine (source so ES_PID is visible to cleanup trap)
source "$(dirname "$0")/start_elasticsearch.sh"

# Start reasoning model server
source "$(dirname "$0")/start_reasoning_model.sh"

# Start check model server
source "$(dirname "$0")/start_check_model.sh"

# Run the check think online script
echo "Start running check think offline"
echo "Project root: ${PROJECT_ROOT}, VLLM API Key: ${VLLM_API_KEY}, Reasoning base url: ${REASONING_BASE_URL}, Check base url: ${CHECK_BASE_URL}"

python -u $PROJECT_ROOT/src/tasks/check_think_offline.py \
    --reasoning_model_path $REASONING_MODEL_PATH \
    --check_model_path $CHECK_MODEL_PATH \
    --dataset_name $DATASET_NAME \
    --dataset_file $DATASET_FILE \
    --retrieval_method $RETRIEVAL_METHOD \
    --embedding_model $EMBEDDING_MODEL \
    --corpus_path $CORPUS_PATH \
    --vector_path $VECTOR_PATH \
    --TopK $TopK \
    --evaluate
    
echo "Check think offline finished."

echo "Shutting down the reasoning model server..."
kill $REASONING_PID
echo "Reasoning model server shut down."

echo "Shutting down the check model server..."
kill $CHECK_PID
echo "Check model server shut down."

echo "Shutting down the elasticsearch engine..."
kill $ES_PID
echo "Elasticsearch engine shut down."

trap - EXIT