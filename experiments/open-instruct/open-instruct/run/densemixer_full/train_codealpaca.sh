# Below is a list of the command-line arguments used in the script:
# - --task (codealpaca, gsm, intent, law, summary, translation)
# - --model (olmoe)
# - --num_gpus
# - --devices (cuda visible devices)
# - --port (deepspeed port)
# - --output_suffix (the output directory will be output/{task}/{model}_{finetuning method}{suffix})
# - --user (experiment personal prefix)
# - --per_device_train_batch_size
# - --per_device_eval_batch_size
# - --total_batch_size
# - --max_seq_length
# - --lr
# - --lr_type (learning rate scheduler type)
# - --warmup_ratio
# - --weight_decay
# - --num_train_epochs
# - --cache_dir
# - --freeze_gate (False, True)
# - --lora_router (frozen, unfrozen)

# For lora, we also have
# - --lora_rank
# - --lora_alpha
# - --lora_dropout

# For esft, we also have
# - --expert_type (gate, token)
export DENSEMIXER_ENABLED=1
export DENSEMIXER_QWEN3=0
export DENSEMIXER_QWEN2=0
export DENSEMIXER_OLMOE=1
export DENSEMIXER_TOPK_MODE=sample_topk
export DENSEMIXER_IMPLEMENTATION=dense_mixer
export WANDB_API_KEY="1532edc16234575030f74f9a5edbfa977ec1ee4b"

# Vanilla Router-Full
bash scripts/train/finetune/full.sh \
    --task codealpaca \
    --model olmoe \
    --total_batch_size 256 \
    --num_train_epochs 2 \
    --num_gpus 8 \
    --devices 0,1,2,3,4,5,6,7 \
    --port 29003 \
    --lr 1e-6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_checkpointing false