export DENSEMIXER_ENABLED=1
export DENSEMIXER_QWEN3=0
export DENSEMIXER_QWEN2=0
export DENSEMIXER_OLMOE=1
export DENSEMIXER_TOPK_MODE=sample_topk
export CUDA_VISIBLE_DEVICES=3

python run_eval_gsm.py \
    --dataset_name RoxanneWsyw/gsm\
    --test_file test.jsonl \
    --save_dir olmoe_results/gsm/sptopk1-10/sptopk-4 \
    --model_name_or_path /home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/open-instruct/open-instruct/output/gsm/olmoe-sptopk1-10 \
    --tokenizer_name_or_path /home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/open-instruct/open-instruct/output/gsm/olmoe-sptopk1-10 \
    --eval_batch_size 64 \
