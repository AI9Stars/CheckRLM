#!/bin/bash

set -e

# Load the config file
source config.sh
export PROJECT_ROOT=$PROJECT_ROOT
export CHECK_MODEL_PATH=$CHECK_MODEL_PATH
export DPO_OUTPUT_DIR=$DPO_OUTPUT_DIR
export DPO_INPUT_FILE=$DPO_INPUT_FILE
export NCCL_P2P_LEVEL=NVL

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun  --nnodes=1 --nproc_per_node=8 --master_addr localhost --master_port 7428 --node_rank 0 $PROJECT_ROOT/src/tasks/train_dpo.py \
    --model_name_or_path $CHECK_MODEL_PATH \
    --dataset_name json \
    --max_length 8192 \
    --max_prompt_length 8000 \
    --output_dir $DPO_OUTPUT_DIR \
    --save_steps 50 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-7 \
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_dir $DPO_OUTPUT_DIR \
    --bf16 True \
    --num_train_epochs 1 \
    --report_to "tensorboard" \
    --save_only_model \
    --gradient_checkpointing \
    --deepspeed $PROJECT_ROOT/src/config/ds_config_zero3.json