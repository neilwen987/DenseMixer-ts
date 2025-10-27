export DENSEMIXER_ENABLED=1
export DENSEMIXER_QWEN3=0
export DENSEMIXER_QWEN2=0
export DENSEMIXER_OLMOE=0
export DENSEMIXER_GPT_OSS=1
# export DENSEMIXER_TOPK_MODE=sample_topk
accelerate launch --num_processes 1 -m lm_eval \
  --model hf \
  --model_args pretrained=openai/gpt-oss-20b,parallelize=True,device_map=auto,offload_folder=/dev/shm/offload,dtype=bfloat16 \
  --tasks gsm8k_cot_zeroshot \
  --batch_size 32 \
  --output_path ./test.json