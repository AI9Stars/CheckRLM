#!/bin/bash

# Load the config file
source config.sh
source check_vllm_service_ready.sh

# Start the reasoning model server
echo "Starting the reasoning model server..."
CUDA_VISIBLE_DEVICES=0,1 vllm serve $REASONING_MODEL_PATH \
    --port $REASONING_PORT \
    --dtype $REASONING_DTYPE \
    --tensor-parallel-size $REASONING_TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $REASONING_GPU_MEMORY_UTILIZATION \
    --seed $SEED \
    --max_model_len 32768 \
    --chat-template "${PROJECT_ROOT}/src/template_qwq.jinja" \
    --enforce-eager &
REASONING_PID=$!
echo "Reasoning model ${REASONING_MODEL_PATH} server started with PID ${REASONING_PID}."

# Wait for the reasoning model server to be ready
if ! check_vllm_service_ready $REASONING_PORT; then
    echo "Failed to start reasoning model server. Exiting..."
    kill $REASONING_PID 
    exit 1
fi