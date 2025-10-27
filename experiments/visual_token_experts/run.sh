export DENSEMIXER_ENABLED=1
export DENSEMIXER_QWEN3=0
export DENSEMIXER_QWEN2=1
export DENSEMIXER_OLMOE=0
export DENSEMIXER_TOPK_MODE=sample_topk
export DENSEMIXER_IMPLEMENTATION=dense_mixer
export CUDA_VISIBLE_DEVICES=0

# python demo.py
python demo_entropy_expert.py