# tools/routing_metrics_collector.py
import os
import torch
import torch.nn.functional as F
from transformers.generation.stopping_criteria import StoppingCriteria
from densemixer.models.qwen2_moe_custom import get_routing_cache_copy
from typing import List, Optional, Union

class RoutingMetricsCollector(StoppingCriteria):
    """
    一个高效的、支持批处理的指标收集器，并可选地作为 StoppingCriteria。
    功能如下：
    1. (Collecting) 在每一步生成时，高效地为批次中的每个序列收集 token_entropy 和 avg_exp_num。
    2. (Stopping) (可选) 当检测到停止字符串时，停止生成。
    3. (Saving) 提供方法，用于在生成结束后一次性保存收集到的所有指标。
    """
    def __init__(self, tokenizer: Optional = None, stop_str: Optional[Union[str, List[str]]] = None, every: int = 1):
        # 如果提供了 stop_str，则必须提供 tokenizer
        if stop_str is not None:
            assert tokenizer is not None, "Tokenizer must be provided when using stop_str."
        
        self.tokenizer = tokenizer
        self.every = max(int(every), 1)
        
        if isinstance(stop_str, str):
            self.stop_sequences = [stop_str]
        else:
            self.stop_sequences = stop_str

        # 初始化用于高效存储数据的列表
        self.all_avg_exp_nums: List[float] = []
        self.all_entropies: List[float] = []
        self.step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: Optional[torch.FloatTensor] = None, **kwargs) -> bool:
        self.step += 1

        # --- 1. 指标收集逻辑 (完全批处理) ---
        if self.step % self.every == 0:
            batch_size = input_ids.shape[0]
            
            # -- 计算 avg_exp_num (批处理) --
            cache = get_routing_cache_copy()
            routing_last_tensors = [v[:, -1, :] for k, v in cache.items() if v.size(1) >= 1]
            
            avg_exp_num_batch = torch.zeros(batch_size, device=input_ids.device)
            if routing_last_tensors:
                # 堆叠所有层的路由权重张量: [num_layers, B, num_experts]
                stacked_routing = torch.stack(routing_last_tensors, dim=0)
                
                # 计算每个序列、每个层激活的专家数: [num_layers, B]
                experts_per_layer = (stacked_routing > 0).sum(dim=2)
                
                # 计算每个序列总共激活的专家数: [B]
                total_experts_activated = experts_per_layer.sum(dim=0)
                
                # 计算每个序列的平均激活专家数: [B]
                avg_exp_num_batch = total_experts_activated / len(routing_last_tensors)
            
            # -- 计算 token_entropy (批处理) --
            entropy_batch = torch.zeros(batch_size, device=input_ids.device)
            if scores is not None:
                last_scores = scores[-1] if isinstance(scores, (list, tuple)) and len(scores) > 0 else scores
                if isinstance(last_scores, torch.Tensor):
                    probs = F.softmax(last_scores, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    entropy_batch = dist.entropy() # Shape: [B]

            # --- 修改部分开始 ---
            # 只有当 entropy > 2 时，才保存对应的 entropy 和 exp_num
            
            # 1. 创建一个布尔掩码，标记出 entropy > 2 的位置
            mask = (entropy_batch >1) & (entropy_batch < 2)
            
            # 2. 使用掩码从批次中筛选出符合条件的 entropy 和 avg_exp_num
            filtered_entropies = entropy_batch[mask]
            filtered_exp_nums = avg_exp_num_batch[mask]
            
            # 3. 将筛选后的数据追加到历史记录中
            if filtered_entropies.numel() > 0: # 确保有数据需要添加
                self.all_avg_exp_nums.extend(filtered_exp_nums.cpu().tolist())
                self.all_entropies.extend(filtered_entropies.cpu().tolist())
            # --- 修改部分结束 ---

            # 打印逻辑保持不变，用于监控每个步骤的原始数据（批次中的第一个样本）
            print(
                f'Step: {self.step}, '
                f'Avg_exp_num (sample 0): {avg_exp_num_batch[0].item():.2f}, '
                f'Token_entropy (sample 0): {entropy_batch[0].item():.2f}'
            )

        # --- 2. 停止逻辑 (保持不变) ---
        if self.stop_sequences is None:
            return False

        # batch_decode 已经是批处理操作
        decoded_texts = self.tokenizer.batch_decode(input_ids)
        # 遍历检查是必要的，因为停止条件可能在不同序列的不同位置触发
        for text in decoded_texts:
            answer_pos = text.rfind("Answer:")
            if answer_pos != -1:
                search_area = text[answer_pos:]
                for stop_word in self.stop_sequences:
                    if stop_word in search_area:
                        print(f"\n--- Stopping generation: Found sequence '{stop_word}' ---\n")
                        return True
        return False

    def save_log(self, save_path: str):
        """
        将收集到的所有指标保存到一个 .pt 文件中。
        保存的数据是一个字典，包含 'avg_exp_num' 和 'token_entropy' 两个键。
        
        Args:
            save_path (str): 完整的文件路径，例如 '/path/to/log.pt'
        """
        if not save_path.endswith('.pt'):
            save_path += '.pt'
            
        # 确保保存路径的目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 将数据整理成一个字典
        data_to_save = {
            'avg_exp_num': self.all_avg_exp_nums,
            'token_entropy': self.all_entropies
        }
        
        torch.save(data_to_save, save_path)
        print(f"--- Metrics log with {len(self.all_avg_exp_nums)} entries saved to: {save_path} ---")

    def reset(self):
        """重置内部状态，以便在新的生成任务中重用同一个实例。"""
        self.all_avg_exp_nums = []
        self.all_entropies = []
        self.step = 0