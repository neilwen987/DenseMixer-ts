export DENSEMIXER_ENABLED=1
export DENSEMIXER_QWEN3=0
export DENSEMIXER_QWEN2=0
export DENSEMIXER_OLMOE=1
export DENSEMIXER_TOPK_MODE=sample_topk
export CUDA_VISIBLE_DEVICES=4 

python run_eval_gsm.py \
    --dataset_name RoxanneWsyw/gsm\
    --test_file test.jsonl \
    --save_dir gsm/original_results/ \
    --model_name_or_path /home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/open-instruct/open-instruct/output/gsm/olmoe-sptopk1-10 \
    --tokenizer_name_or_path /home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/open-instruct/open-instruct/output/gsm/olmoe-sptopk1-10 \ 
    --eval_batch_size 64