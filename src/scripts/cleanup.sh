#!/bin/bash

# Function to clean up all serves if fails
cleanup() {
    echo "Caught an error or exit signal. Cleaning up..."
    if [ -n "$REASONING_PID" ]; then
        echo "Killing reasoning model server with PID $REASONING_PID..."
        kill $REASONING_PID 2>/dev/null || true
    fi
    if [ -n "$CHECK_PID" ]; then
        echo "Killing check model server with PID $CHECK_PID..."
        kill $CHECK_PID 2>/dev/null || true
    fi
    if [ -n "$ES_PID" ]; then
        echo "Killing Elasticsearch engine with PID $ES_PID..."
        kill $ES_PID 2>/dev/null || true
    fi
    echo "Cleanup complete."
    exit 1
}