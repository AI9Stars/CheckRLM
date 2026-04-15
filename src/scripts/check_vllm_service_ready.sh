#!/bin/bash

# Function to check if a service is ready
check_vllm_service_ready() {
    local port=$1
    local max_retries=20  
    local retry_interval=30  
    local retry_count=0

    echo "Checking if service on port ${port} is ready..."
    while [ $retry_count -lt $max_retries ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null; then
            echo "Service on port ${port} is ready!"
            return 0
        else
            echo "Service on port ${port} not ready yet, retrying in ${retry_interval} seconds..."
            sleep $retry_interval
            retry_count=$((retry_count + 1))
        fi
    done

    echo "Service on port ${port} failed to start within the expected time."
    return 1
}