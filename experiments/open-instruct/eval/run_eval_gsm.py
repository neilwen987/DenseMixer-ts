import argparse
import os
import re
import json
import random
import torch
import evaluate
from utils import (
    generate_completions,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    dynamic_import_function,
    ensure_dir
)
from datasets import load_dataset
import pandas as pd

exact_match = evaluate.load("exact_match")

def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output

def main(args):
    random.seed(42)
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    if args.use_batchtopk:
        import os
        os.environ['DENSEMIXER_ENABLED'] = '1'
        os.environ['DENSEMIXER_OLMOE'] = '1'
        os.environ['DENSEMIXER_TOPK_MODE'] = 'batch_topk'
        import densemixer
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

    # 检查forward方法是否已被替换
    print(OlmoeSparseMoeBlock.forward.__module__)
    # 如果patch成功，应该显示densemixer相关的模块名
    print("Loading data...")
    test_data = []

    test_dataset = load_dataset(args.dataset_name, data_files=args.test_file)
    test_df = pd.DataFrame(test_dataset['train'])

    test_data = []

    if "question" in test_df.columns and "answer" in test_df.columns:
        for _, row in test_df.iterrows():
            test_data.append({
                "question": row["question"],
                "answer": row["answer"].split("####")[1].strip() if "####" in row["answer"] else row["answer"].strip()
            })

    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    if args.max_examples and len(test_data) > args.max_examples:
        # test_data = random.sample(test_data, args.max_examples)
        test_data = test_data[:args.max_examples]

    ensure_dir(args.save_dir)

    prompt_prefix = "Answer the following question.\n\n"
    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    
    for example in test_data:
        prompt = prompt_prefix + "Question: " + example["question"].strip()
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "Answer:"
            else:
                prompt += " Answer:"
        else:
            prompt += "\nAnswer:"
        prompts.append(prompt)
    
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    elif args.base_model_name_or_path:
        model, tokenizer = load_dexperts_model_and_tokenizer(
            args.base_model_name_or_path,
            args.expert_model_name_or_path,
            chat_response_prefix="Answer:",
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=512,
        batch_size=args.eval_batch_size,
        do_sample=False
    )

    outputs = [trim_output(o) for o in outputs]

    predictions = []
    for output in outputs:
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)

    print("Calculating accuracy...")
    targets = [example["answer"] for example in test_data]

    em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match : {em_score}")

    predictions = [{
        "question": example["question"],
        "answer": example["answer"],
        "model_output": output,
        "prediction": pred
    } for example, output, pred in zip(test_data, outputs, predictions)]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump({
            "exact_match": em_score
        }, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--use_batchtopk", type=bool, default=False)
    parser.add_argument("--max_examples", type=int, default=None, help="maximum number of examples to evaluate.")
    parser.add_argument("--save_dir", type=str, default="results/gsm")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model path.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Tokenizer path.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="Use slow tokenizer.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit mode.")
    parser.add_argument("--base_model_name_or_path", type=str, default='meta-llama/Llama-2-13b-hf')
    parser.add_argument("--expert_model_name_or_path", type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--use_chat_format", action="store_true", help="Use chat format.")
    parser.add_argument("--chat_formatting_function", type=str, default="eval.templates.create_prompt_with_tulu_chat_format")
    args = parser.parse_args()
    main(args)
