 python run_eval_gsm.py \
    --dataset_name RoxanneWsyw/gsm\
    --test_file test.jsonl \
    --save_dir gsm/original_results/ \
    --model_name_or_path allenai/OLMoE-1B-7B-0125 \
    --tokenizer_name_or_path allenai/OLMoE-1B-7B-0125 \
    --max_examples 100 \
    --eval_batch_size 256