#!/bin/bash

# Debug demo.py 启动脚本
# 设置必要的环境变量

echo "=== 启动 Debug Demo ==="
echo "当前时间: $(date)"
echo "当前目录: $(pwd)"

# 设置 CUDA 环境
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1  # 启用 CUDA 同步，便于调试

# 设置 DenseMixer 相关环境变量
export DENSEMIXER_ENABLED=1
export DENSEMIXER_QWEN3=0
export DENSEMIXER_QWEN2=1
export DENSEMIXER_OLMOE=0
export DENSEMIXER_TOPK_MODE=sample_topk

# 设置 Python 路径
export PYTHONPATH="/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts:${PYTHONPATH}"

# 设置 HuggingFace 相关
export HF_ALLOW_CODE_EVAL="1"
export HF_TOKEN=""

# 设置 PyTorch 相关
export TORCH_USE_CUDA_DSA=1
export TORCH_CUDNN_V8_API_ENABLED=1

# 创建输出目录
mkdir -p routing_weights/debug_$(date +%Y%m%d_%H%M%S)

echo "=== 环境变量设置完成 ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "DENSEMIXER_ENABLED: $DENSEMIXER_ENABLED"
echo "DENSEMIXER_QWEN2: $DENSEMIXER_QWEN2"
echo "PYTHONPATH: $PYTHONPATH"

echo "=== 开始运行 demo.py ==="

# 运行 demo.py，启用详细输出
python -u demo.py 2>&1 | tee debug_run_$(date +%Y%m%d_%H%M%S).log

echo "=== Demo 运行完成 ==="
echo "日志已保存到: debug_run_$(date +%Y%m%d_%H%M%S).log" 