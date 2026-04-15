#!/bin/bash

set -e

# Load the config file
source config.sh
export PROJECT_ROOT=$PROJECT_ROOT
export VLLM_API_KEY=$VLLM_API_KEY 
export REASONING_BASE_URL=$REASONING_BASE_URL

METHOD="NoR" 

# Import functions and scripts
source cleanup.sh

trap cleanup EXIT

# Start reasoning model server (source so REASONING_PID is visible to cleanup trap)
source "$(dirname "$0")/start_reasoning_model.sh"

# Run the script
echo "Start running ${METHOD} script..."
echo "Project root: ${PROJECT_ROOT}, VLLM API Key: ${VLLM_API_KEY}, Reasoning base url: ${REASONING_BASE_URL}"

python -u $PROJECT_ROOT/src/tasks/base.py \
    --reasoning_model_path $REASONING_MODEL_PATH \
    --method $METHOD \
    --dataset_name $DATASET_NAME \
    --dataset_file $DATASET_FILE \
    --TopK $TopK \
    --evaluate
    
echo "${METHOD} script finished."

echo "Shutting down the reasoning model server..."
kill $REASONING_PID
echo "Reasoning model server shut down."

trap - EXIT