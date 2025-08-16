# tools/routing_stopping.py
import os
import torch
import torch.nn.functional as F
from transformers.generation.stopping_criteria import StoppingCriteria
from densemixer.models.qwen2_moe_custom import get_routing_cache_copy
from typing import List, Optional, Union

class RoutingAlignDump(StoppingCriteria):
    """
    一个集成的 StoppingCriteria，功能如下：
    1. (Dumping) 在每一步生成时，收集新生成的 token、路由权重等信息。
    2. (Stopping) (可选) 当在生成的内容中检测到指定的停止字符串时，停止整个生成过程。
    3. (Saving) 提供一个方法，用于在生成结束后一次性保存所有收集到的信息。
    """
    def __init__(self, tokenizer, stop_str: Optional[Union[str, List[str]]] = None, every=1):
        self.tokenizer = tokenizer
        self.every = max(int(every), 1)
        
        # --- 新增：处理停止字符串 ---
        if isinstance(stop_str, str):
            self.stop_sequences = [stop_str]
        else:
            self.stop_sequences = stop_str

        # --- 修改：初始化用于收集数据的内部列表和状态 ---
        self.history = []
        self.step = 0
        self.prompt_len = None

    def __call__(self, input_ids, scores=None, **kwargs) -> bool:
        if self.prompt_len is None:
            self.prompt_len = input_ids.shape[1] - 1

        self.step += 1

        # --- Dumping Logic (现在是收集逻辑) ---
        if self.step % self.every == 0:
            token_ids = input_ids[:, -1].detach().cpu()
            tokens = [self.tokenizer.decode([tid.item()], skip_special_tokens=False) for tid in token_ids]

            cache = get_routing_cache_copy()
            routing_last = {str(k): v[:, -1, :].cpu() for k, v in cache.items() if v.size(1) >= 1}
            avg_exp_num = 0
            if routing_last:
                for key in routing_last:
                    avg_exp_num += (routing_last[key] > 0).sum().item()
                avg_exp_num /= len(routing_last)

            payload = {
                "step": self.step,
                "prompt_len": self.prompt_len,
                "token_ids": token_ids,
                "tokens": tokens,
                "routing_last": routing_last,
                "avg_exp_num": avg_exp_num,
            }
            
            entropy = None
            if scores is not None:
                last_scores = scores[-1] if isinstance(scores, (list, tuple)) and len(scores) > 0 else scores
                if isinstance(last_scores, torch.Tensor):
                    probs = F.softmax(last_scores, dim=-1).detach().cpu()
                    dist = torch.distributions.Categorical(probs=probs)
                    entropy = dist.entropy() # Shape: [B]
                    payload["token_entropy"] = entropy.tolist() # 保存为 list
            
            # --- 核心修改：不再保存文件，而是追加到 history 列表 ---
            self.history.append(payload)

            # 打印逻辑可以保留，用于实时监控
            entropy_str = f", Token Entropy: {entropy[0]:.2f}" if entropy is not None and entropy.numel() == 1 else ""
            print(f'Gen : {tokens}, Avg_exp_num: {avg_exp_num:.2f}{entropy_str}')

        # --- Stopping Logic (保持不变) ---
        if self.stop_sequences is None:
            return False

        decoded_texts = self.tokenizer.batch_decode(input_ids)
        for text in decoded_texts:
            answer_pos = text.rfind("Answer:")
            if answer_pos != -1:
                search_area = text[answer_pos:]
                for stop_word in self.stop_sequences:
                    if stop_word in search_area:
                        print(f"\n--- Stopping generation: Found sequence '{stop_word}' ---\n")
                        return True
        return False

    # --- 新增：保存所有收集数据的公共方法 ---
    def save_log(self, save_path):
        """
        将收集到的所有生成步骤信息保存到一个 .pt 文件中。
        Args:
            save_path (str): 完整的文件路径，例如 '/path/to/dir/generation_log.pt'
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.history, save_path)
        print(f"--- Generation log with {len(self.history)} steps saved to: {save_path} ---")

    # --- 新增：重置方法，方便在循环中使用 ---
    def reset(self):
        self.history = []
        self.step = 0
        self.prompt_len = None