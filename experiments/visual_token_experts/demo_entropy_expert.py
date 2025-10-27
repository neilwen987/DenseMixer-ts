# scripts/demo_batch_generate_chunked.py
import os
import torch
import math # 用于计算总批次数
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
# 假设您的自定义文件在同一目录下或在PYTHONPATH中
from routing_stopping_entropy_expert import RoutingMetricsCollector 
from datasets import load_dataset, get_dataset_config_names

# --- 模拟 process_docs 的功能 ---
def process_docs(dataset):
    return dataset.filter(lambda example: 'problem' in example)

os.environ['HF_TOKEN']=""

# --- 配置参数 ---
MODEL_PATH = '/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/llama-factory/LLaMA-Factory/saves/qwen1.5-moe-a2.7b/full/sptopk/sft_bs64/qwen1.5_gsm_lr1e-6-topk6'
BATCH_SIZE = 8

# 1. 加载并处理数据集
print("Loading and processing dataset...")
dataset_name = "EleutherAI/hendrycks_math"
subset_names = get_dataset_config_names(dataset_name)

task = subset_names[4] # 假设选择 'prealgebra'
math_dataset = load_dataset("EleutherAI/hendrycks_math", task)['test']
math_dataset = process_docs(math_dataset)

SAVE_DIR = f'/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/visual_token_experts/routing_weights/{task}'
os.makedirs(SAVE_DIR, exist_ok=True) # 直接在这里创建目录更稳妥

TOTAL_SAMPLES = len(math_dataset)

# 2. 定义 Prompt 模板并格式化所有样本
def format_prompt(item):
    return f"Problem: {item['problem']}\nAnswer:"

all_prompts = [format_prompt(item) for item in math_dataset]
print(f"Prepared a total of {len(all_prompts)} prompts.")
print("-" * 20)

# 3. 加载模型和 Tokenizer
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# 4. 准备指标收集器和日志
routing_collector = RoutingMetricsCollector(tokenizer=tokenizer, stop_str="Problem:", every=1)
all_generated_texts = [] 

# 5. 按批次循环处理所有 prompts
num_batches = math.ceil(TOTAL_SAMPLES / BATCH_SIZE)
print(f"Total samples: {TOTAL_SAMPLES}, Batch size: {BATCH_SIZE}. Will run {num_batches} batches.")

# --- 新增：用于分块保存的文件计数器 ---
chunk_counter = 1

for i in range(num_batches):
    print(f"\n--- Processing Batch {i+1}/{num_batches} ---")
    
    start_index = i * BATCH_SIZE
    end_index = min((i + 1) * BATCH_SIZE, TOTAL_SAMPLES)
    batch_prompts = all_prompts[start_index:end_index]
    
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    routing_collector.reset()
    criteria = StoppingCriteriaList([routing_collector])

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=criteria,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )

    batch_generated_texts = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
    all_generated_texts.extend(batch_generated_texts)
    
    print("Example output from this batch:")
    print(batch_generated_texts[0])

    # --- 关键改动：每 50 个批次保存一次 ---
    # (i + 1) 是当前处理完的批次数。当它能被 50 整除时，就触发保存。
    # 同时要确保不是最后一个批次，避免与最终的保存重复。
    if (i + 1) % 50 == 0 and (i + 1) < num_batches:
        print(f"\n--- Reached batch {i+1}. Saving intermediate metrics... ---")
        # 定义一个带编号的文件名
        output_log_path = os.path.join(SAVE_DIR, f'generation_metrics_chunk_0.5_2_{chunk_counter}.pt')
        routing_collector.save_log(output_log_path)
        
        # 重置收集器，为下一个 50 批次的数据做准备
        routing_collector.reset()
        print(f"--- Metrics collector has been reset. ---")
        
        # 更新文件计数器
        chunk_counter += 1


# 6. 所有批次处理完毕后，保存最后剩余的指标
print("\n--- All batches processed. Saving final metrics... ---")
# 检查收集器中是否还有未保存的数据
if routing_collector.all_entropies:
    output_log_path = os.path.join(SAVE_DIR, f'generation_metrics_chunk_{chunk_counter}.pt')
    routing_collector.save_log(output_log_path)
    print(f"Final metrics for the last set of samples saved to: {output_log_path}")
else:
    print("No new metrics to save in the final step (total batches might be a multiple of 50).")


# 7. (可选) 打印所有结果
# 注意：这一步仍然会等待所有批次完成后才执行
print("\n--- All Generated Texts ---")
for i, text in enumerate(all_generated_texts):
    print(f"--- Result for Sample {i+1} ---")
    print(text)
    print("-" * 20)