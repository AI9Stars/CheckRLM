#!/bin/bash

# Load the config file
source config.sh
source check_vllm_service_ready.sh

# Start the check model server
echo "Starting the check model server..."
CUDA_VISIBLE_DEVICES=2 vllm serve $CHECK_MODEL_PATH \
    --port $CHECK_PORT \
    --dtype $CHECK_DTYPE \
    --tensor-parallel-size $CHECK_TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $CHECK_GPU_MEMORY_UTILIZATION \
    --seed $SEED \
    --enable-prefix-caching \
    --enforce-eager &
CHECK_PID=$!
echo "Check model ${CHECK_MODEL_PATH} server started with PID ${CHECK_PID}."

# Wait for the check model server to be ready
if ! check_vllm_service_ready $CHECK_PORT; then
    echo "Failed to start check model server. Exiting..."
    kill $REASONING_PID 
    kill $CHECK_PID     
    exit 1
fi