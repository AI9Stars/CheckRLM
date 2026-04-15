#!/bin/bash

# Load the config file
source config.sh

# Start Elasticsearch engine
echo "Starting the Elasticsearch engine..."
_prev_es_pwd="$(pwd)"
cd "$ES_PATH" || { echo "Cannot cd to ES_PATH=$ES_PATH"; exit 1; }
ES_JAVA_OPTS="-Xms2g -Xmx2g" ./bin/elasticsearch &
ES_PID=$!
cd "$_prev_es_pwd" || true
echo "Elasticsearch server started with PID ${ES_PID}."