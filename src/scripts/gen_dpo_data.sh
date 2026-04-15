#!/bin/bash

set -e

# Load the config file
source config.sh
export PROJECT_ROOT=$PROJECT_ROOT
export VLLM_API_KEY=$VLLM_API_KEY 
export REASONING_BASE_URL=$REASONING_BASE_URL
export CHECK_BASE_URL=$CHECK_BASE_URL
export OPENAI_API_KEY=$OPENAI_API_KEY
export OPENAI_BASE_URL=$OPENAI_BASE_URL

EVALUATE_MODEL_NAME="gpt-4o-mini"
DPO_NUMBER=2500

# Import functions and scripts
source cleanup.sh

trap cleanup EXIT

# Start elasticsearch engine 
source "$(dirname "$0")/start_elasticsearch.sh"

# Start reasoning model server
source "$(dirname "$0")/start_reasoning_model.sh"

# Start check model server
source "$(dirname "$0")/start_check_model.sh"

# Run the DPO Training Data Generation script
echo "Start running DPO Training Data Generation"
echo "Project root: ${PROJECT_ROOT}, VLLM API Key: ${VLLM_API_KEY}, Reasoning base url: ${REASONING_BASE_URL}, Check base url: ${CHECK_BASE_URL}, OpenAI API Key: ${OPENAI_API_KEY}, OpenAI base url: ${OPENAI_BASE_URL}"

python -u $PROJECT_ROOT/src/tasks/generate_dpo_data.py \
    --reasoning_model_path $REASONING_MODEL_PATH \
    --check_model_path $CHECK_MODEL_PATH \
    --evaluate_model $EVALUATE_MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --dataset_file $DATASET_FILE \
    --dpo_number $DPO_NUMBER \
    --max_steps 3 \
    --TopK $TopK
    
echo "DPO Training Data Generation finished."

# Shut down the reasoning and check model servers
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