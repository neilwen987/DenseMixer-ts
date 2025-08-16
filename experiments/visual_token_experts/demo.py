
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from densemixer.models.qwen2_moe_custom import clear_routing_cache
from routing_stopping import RoutingAlignDump
from datasets import load_dataset
from utils import process_docs,process_results
import os
import pandas as pd



# 1. 加载并处理 hendrycks_math 数据集

math_dataset = load_dataset("EleutherAI/hendrycks_math", "algebra")['test']
math_dataset = process_docs(math_dataset)

# 假设 process_docs 已经处理好了数据集，我们取一部分作为示例
# 如果 process_docs 没有进行模板化，我们将在这里进行
# 我们选择一小部分数据进行演示，例如前4条
indices_to_select = [1,2,3,4,5,10 ,16,48,72,47]
# 2. 使用 .select() 方法从数据集中挑选出这些问题

selected_dataset = math_dataset.select(indices_to_select)

# 2. 定义 Prompt 模板并格式化数据集

def format_prompt(item):
    """将数据集中的每一项格式化为指定的模板"""
    return f"Problem: {item['problem']}\nAnswer:"

# 将模板应用到我们的数据子集上

prompts = [format_prompt(item) for item in selected_dataset]
print("Formatted Prompts:")
for p in prompts:
    print(p)
    print("-" * 20)

# 3. 加载模型和 Tokenizer

model_path = '/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/llama-factory/LLaMA-Factory/saves/qwen1.5-moe-a2.7b/full/sptopk/sft_bs64/base_s1k_bs16_lr1e-5_zero3'
assert model_path, "请设置环境变量 MODEL_PATH 为你的模型目录"


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# -------- 关键：为批处理设置 padding token --------

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # 通常将 pad_token 设置为 eos_token

model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        )

# 4. 批处理 Tokenization
padding=True #会将批次内的所有序列填充到最长序列的长度
# 5. 准备生成参数并执行批处理生成
# 新一轮生成前清缓存

clear_routing_cache()

save_dir = '/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/visual_token_experts/routing_weights/'
routing_dumper = RoutingAlignDump(tokenizer, stop_str="Problem:", every=1)

criteria = StoppingCriteriaList([routing_dumper])
with torch.inference_mode():
# model.generate 现在会处理批次中的所有 prompts
    for idx,prompt in enumerate(prompts):
        routing_dumper.reset()
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=criteria,
            use_cache=True,
            # -------- 关键：在生成时也使用 eos_token_id 作为 pad_token_id --------
            pad_token_id=tokenizer.eos_token_id
            )
        output_log_path = os.path.join(save_dir, f'generation_log_{idx}.pt')
        routing_dumper.save_log(output_log_path)

print("Generated texts:")
generated_texts = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
for text in generated_texts:
    print(text)
    print("-" * 20)



print(f"Routing snapshots saved to: {save_dir}") 