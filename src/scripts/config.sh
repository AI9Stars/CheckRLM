# Project root
PROJECT_ROOT="" #  Absolute path to the project root

# Reasoning Model Parameters
REASONING_MODEL_PATH="" # Path to the reasoning model                 
REASONING_DTYPE="bfloat16"                      # Data type
REASONING_GPU_MEMORY_UTILIZATION=0.9            # GPU memory utilization
REASONING_TENSOR_PARALLEL_SIZE=2                # Tensor parallel size
REASONING_PORT=8001                             # Server port
REASONING_BASE_URL="http://localhost:${REASONING_PORT}/v1"      # Base URL

# Check Model Parameters
CHECK_MODEL_PATH="" # Path to the check model                       
CHECK_DTYPE="bfloat16"                      # Data type
CHECK_GPU_MEMORY_UTILIZATION=0.9            # GPU memory utilization
CHECK_TENSOR_PARALLEL_SIZE=1                # Tensor parallel size
CHECK_PORT=8002                             # Server port
CHECK_BASE_URL="http://localhost:${CHECK_PORT}/v1"      # Base URL

# VLLM Parameters
VLLM_API_KEY="EMPTY" # API key for VLLM
SEED=42

# Dataset Parameters
DATASET_NAME="2wikimultihopqa" # choices=["hotpotqa", "2wikimultihopqa", "musique", "simpleqa", "iirc"]
DATASET_FILE="${PROJECT_ROOT}/data/${DATASET_NAME}/test_subsampled.jsonl"
# DATASET_FILE="${PROJECT_ROOT}/data/2wikimultihopqa/corpus/train.json"
TopK=3 # Retrieval file number

# Elasticsearch Parameter
ES_PATH=""
ES_URL="http://localhost:9200"

# OpenAI Parameters
OPENAI_API_KEY=""
OPENAI_BASE_URL=""

# Embedding Parameters
EMBEDDING_MODEL=""
CORPUS_PATH="${PROJECT_ROOT}/data/${DATASET_NAME}/corpus" # Path to the corpus using for build index
VECTOR_PATH="${PROJECT_ROOT}/data/${DATASET_NAME}/corpus/${DATASET_NAME}_bge-large-en-v1.5.bin" # Path to the vector file

# DPO Parameters
DPO_INPUT_FILE=""
DPO_OUTPUT_DIR="${PROJECT_ROOT}/save_model/Qwen2.5-14B_DPO"