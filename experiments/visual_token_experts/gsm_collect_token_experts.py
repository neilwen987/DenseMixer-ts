import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from densemixer.models.olmoe_custom import clear_routing_cache, get_routing_cache_copy


def format_prompt(problem: str) -> str:
    return f"Problem: {problem}\nAnswer:"


def main(
    model_path: str,
    save_dir: str,
    limit: int = 100,
    split: str = "test",
    hf_subset: str = "main",
    devices: str = "0",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    os.makedirs(save_dir, exist_ok=True)

    dataset = load_dataset("openai/gsm8k", hf_subset)[split]

    prompts: List[str] = []
    for i, item in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        prompts.append(format_prompt(item["question"]))

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # 仅前向一次，使用 OLMoE 缓存拿到 [B, S, E]，不生成任何新 token
    with torch.inference_mode():
        for idx, prompt in enumerate(prompts):
            clear_routing_cache()
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
            _ = model(**inputs, use_cache=False)

            cache = get_routing_cache_copy()  # {layer_id: Tensor[B, S, E]}
            if not cache:
                print(f"No routing cache for sample {idx}. Skipped.")
                continue

            # 堆叠层，统计每个 token 的平均激活专家数
            layers = []
            for _, v in cache.items():
                # v: [B, S, E], 取 B=1
                layers.append((v[0] > 0).sum(dim=-1).cpu())  # [S]
            stacked = torch.stack(layers, dim=0)  # [L, S]
            avg_exp_per_token = stacked.float().mean(dim=0)  # [S]

            token_ids = inputs.input_ids[0].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            # 保存为可视化脚本期望的逐 token 条目（每条只包含一个 token 及其 avg_exp_num）
            history_like = []
            for step_i, (tid, tok, avgv) in enumerate(zip(token_ids, tokens, avg_exp_per_token.tolist())):
                history_like.append({
                    "step": step_i,
                    "prompt_len": len(token_ids),
                    "token_ids": [tid],
                    "tokens": [tok],
                    "routing_last": None,
                    "avg_exp_num": torch.tensor(avgv, dtype=torch.float32),
                    "token_entropy": None,
                })

            out_path = os.path.join(save_dir, f"gsm_token_experts_{idx}.pt")
            torch.save(history_like, out_path)
            print(f"Saved prompt-only token experts to: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--hf_subset", type=str, default="main")
    parser.add_argument("--devices", type=str, default="0")
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        save_dir=args.save_dir,
        limit=args.limit,
        split=args.split,
        hf_subset=args.hf_subset,
        devices=args.devices,
    )


