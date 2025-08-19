#!/bin/bash
export DENSEMIXER_ENABLED=1
export DENSEMIXER_QWEN3=0
export DENSEMIXER_QWEN2=1
export DENSEMIXER_OLMOE=0
export DENSEMIXER_TOPK_MODE=sample_topk
export DENSEMIXER_IMPLEMENTATION=dense_mixer
# create log dir (if not exist)
LOGS_DIR="logs"
mkdir -p $LOGS_DIR

# define config name as variable
method="full"
# list of config files to run
CONFIG_FILES=(
    "base_s1k_bs16_lr1e-5_zero3.yaml"
)

export WANDB_API_KEY="1532edc16234575030f74f9a5edbfa977ec1ee4b"
export DISABLE_VERSION_CHECK=1
export WANDB_PROJECT="MoE-Finetune-qwen1.5-v2"

# loop through config files
for config_file in "${CONFIG_FILES[@]}"; do
    # create log name without .yaml extension
    LOG_FILE="$LOGS_DIR/train_${method}_${config_file%.*}.log"
    
    # print starting info
    echo "========================================"
    echo "start training: $(date)"
    echo "config: $config_file"
    echo "log saved at: $LOG_FILE"
    
    # define the command
    CMD="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/qwen1.5moe/${method}/sptopk/${config_file}"
    
    # log the command
    echo "executing command: $CMD" | tee $LOG_FILE
    echo "----------------------------------------" | tee -a $LOG_FILE
    
    # execute the command and append to log file
    eval "$CMD" 2>&1 | tee -a $LOG_FILE
    
    echo "completed training for: $config_file"
    echo "========================================"
done

echo "All training jobs completed at: $(date)"
