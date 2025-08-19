#!/bin/bash
export DENSEMIXER_ENABLED=1
export DENSEMIXER_QWEN3=0
export DENSEMIXER_QWEN2=1
export DENSEMIXER_OLMOE=0
export DENSEMIXER_TOPK_MODE=topk
export DENSEMIXER_IMPLEMENTATION=dense_mixer
# create log dir (if not exist)
LOGS_DIR="logs"
mkdir -p $LOGS_DIR

# define config name as variable
method="full"

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 从脚本目录计算到配置文件的路径
CONFIG_BASE_DIR="$SCRIPT_DIR/../../../examples/train_full/moonightmoe"
# LLaMA-Factory 根目录
LLAMA_FACTORY_DIR="$SCRIPT_DIR/../../.."

# list of config files to run

CONFIG_FILES=(
    "moonlight_codealpaca_lr1e-6.yaml"
    "moonlight_esft_law_lr1e-5.yaml"
    "moonlight_esft_summary_lr1e-6.yaml"
    "moonlight_gsm_lr1e-6.yaml"
)


export WANDB_API_KEY="1532edc16234575030f74f9a5edbfa977ec1ee4b"
export WANDB_PROJECT="MoE-Finetune-moonlinght"
export DISABLE_VERSION_CHECK=1

# loop through config files
for config_file in "${CONFIG_FILES[@]}"; do
    # create log name without .yaml extension
    LOG_FILE="$LOGS_DIR/train_${method}_${config_file%.*}.log"
    
    # 构建配置文件的完整路径
    CONFIG_PATH="$CONFIG_BASE_DIR/${method}/${config_file}"
    
    # 检查配置文件是否存在
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "ERROR: Config file not found: $CONFIG_PATH"
        continue
    fi
    
    # print starting info
    echo "========================================"
    echo "start training: $(date)"
    echo "config: $config_file"
    echo "config path: $CONFIG_PATH"
    echo "log saved at: $LOG_FILE"
    
    # 切换到 LLaMA-Factory 目录，使用相对路径来保证 deepspeed 配置正确
    cd "$LLAMA_FACTORY_DIR"
    
    # 使用相对路径构建命令（相对于 LLaMA-Factory 目录）
    CONFIG_REL_PATH="examples/train_full/qwen1.5moe/${method}/${config_file}"
    CMD="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train $CONFIG_REL_PATH"
    
    # log the command
    echo "executing command: $CMD" | tee "$SCRIPT_DIR/$LOG_FILE"
    echo "working directory: $(pwd)" | tee -a "$SCRIPT_DIR/$LOG_FILE"
    echo "----------------------------------------" | tee -a "$SCRIPT_DIR/$LOG_FILE"
    
    # execute the command and append to log file
    eval "$CMD" 2>&1 | tee -a "$SCRIPT_DIR/$LOG_FILE"
    
    echo "completed training for: $config_file"
    echo "========================================"
done

echo "All training jobs completed at: $(date)"
