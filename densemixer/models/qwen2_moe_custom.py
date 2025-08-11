from unittest.util import strclass
import torch
import torch.nn.functional as F
from ..logging_utils import log_custom_forward_usage
from .. import config as densemixer_config
import os  # 新增
# 全局存储每一层的routing_weights历史
_layer_routing_cache = {}
_generation_mode = False  # 标记是否在生成模式
class CustomQwen2MoeSparseMoeBlock:
    @staticmethod
    def forward(self, hidden_states: torch.Tensor):
        """
        Inherited from official OlmoeSparseMoeBlock, implements dense backward functionality:
        Forward output remains the same as official (i.e., sparse computation results),
        but during backward propagation, dense computation gradients are passed back through straight-through gradient,
        dense output is obtained by computing each expert on all tokens and weighted by full routing weights.

        Input:
            hidden_states: Tensor, shape (batch_size, sequence_length, hidden_dim)
        Output:
            final_output: Tensor, shape (batch_size, sequence_length, hidden_dim)
            router_logits: Tensor, shape (batch_size * sequence_length, num_experts)
        """
        log_custom_forward_usage("Qwen2-MoE")
        batch_size, seq_length, hidden_dim = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        flat_hidden = hidden_states.view(-1, hidden_dim)  # (B*seq_len, hidden_dim)
        N_tokens = flat_hidden.size(0)
        layer_id = getattr(self, 'layer_idx', id(self))
        if not hasattr(self, 'i'):
            self.i = 0
        self.i += 1
        
        # Compute routing logic
        router_logits = self.gate(flat_hidden)  # (B*L, num_experts)
        router_logits = router_logits.to(dtype=dtype)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # (B*L, num_experts)

        # Select top-k experts and cast to match input dtype to avoid dtype mismatch on in-place updates
        if densemixer_config.topk_mode == "topk":
            print('Using Topk,this is the {}-th call of this layer'.format(self.i))
            routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        elif densemixer_config.topk_mode == "sample_topk":
            # print('Using SPTopk,this is the {}-th call of this layer'.format(self.i))
            # print('using sample_topk')
            routing_weights_reshaped = routing_weights.view(batch_size, seq_length, -1)  # (N, Seq_length, Expert)
            
            _, top6_indices = torch.topk(routing_weights_reshaped, k=6, dim=-1)
            max_mask = torch.zeros_like(routing_weights_reshaped, dtype=torch.bool).scatter_(-1, top6_indices, True)
            routing_weights_reshaped = routing_weights_reshaped * max_mask.detach()
            
            _, top1_indices = torch.topk(routing_weights_reshaped, k=1, dim=-1)

            _, flat_indices = torch.topk(routing_weights_reshaped.view(batch_size, -1), k=self.top_k * seq_length, dim=1)
            
            select_mask = torch.zeros_like(routing_weights_reshaped.view(batch_size, -1), dtype=torch.bool).scatter_(-1, flat_indices, True).reshape(routing_weights_reshaped.shape)
            select_mask.scatter_(-1, top1_indices, True)
            expert_counts = (select_mask > 0).sum(dim=-1) 
            max_experts_per_token = expert_counts.max()
            filtered_scores = (routing_weights_reshaped * select_mask.detach()).view(-1, self.num_experts)
            routing_weights_topk, selected_experts = torch.topk(filtered_scores, k=max_experts_per_token, dim=1)
            # routing_weights_topk, selected_experts = handle_sample_topk_with_cache(
            #     self, routing_weights, self.top_k, batch_size, seq_length, layer_id, N_tokens
            # )
        routing_weights_topk = routing_weights_topk.to(dtype=dtype)
        if self.norm_topk_prob:
            routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
            routing_weights_topk = routing_weights_topk.to(dtype=dtype)

        # Convert full routing_weights to consistent dtype for dense accumulation
        routing_weights = routing_weights.to(dtype=dtype)
        # Add shared expert contribution to both sparse and dense outputs
        shared_expert_output = self.shared_expert(flat_hidden)  # (N_tokens, hidden_dim)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(flat_hidden)) * shared_expert_output
        # Prepare accumulators: one for dense_outputs, one for sparse_outputs
        dense_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)
        sparse_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)

        # For mapping top-k positions when accumulating sparse_outputs
        # selected_experts: (N_tokens, top_k)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # Compute current expert output for all tokens
            expert_output = expert_layer(flat_hidden)  # (N_tokens, hidden_dim)
            # Register hook for all experts to mask non-selected token gradients
            activation_mask = (selected_experts == expert_idx).any(dim=1).float().unsqueeze(-1).to(dtype)
            if expert_output.requires_grad:
                expert_output.register_hook(lambda grad, mask=activation_mask: grad * mask)
            expert_output = expert_output.to(dtype=dtype)

            # Dense accumulation: multiply by full routing weight and add
            weight_full = routing_weights[:, expert_idx].unsqueeze(-1)  # (N_tokens, 1)
            dense_outputs = dense_outputs + expert_output * weight_full

            # Sparse accumulation: find tokens where this expert is among top_k
            matches = (selected_experts == expert_idx)
            if matches.any():
                token_indices, k_indices = torch.where(matches)
                weights_topk = routing_weights_topk[token_indices, k_indices].unsqueeze(-1)  # (num_matches, 1)
                sparse_outputs[token_indices] = sparse_outputs[token_indices] + expert_output[token_indices] * weights_topk

        sparse_outputs = sparse_outputs + shared_expert_output
        dense_outputs = dense_outputs + shared_expert_output

        # Combine sparse forward output and dense backward output
        final_flat = sparse_outputs.detach() + (dense_outputs - dense_outputs.detach())
        final_flat = final_flat.to(dtype=dtype)
        final_output = final_flat.view(batch_size, seq_length, hidden_dim)

        return final_output, router_logits

def handle_sample_topk_with_cache(moe_block, routing_weights, top_k, batch_size, seq_length, layer_id, N_tokens):
    """处理带缓存的sample_topk"""
    global _layer_routing_cache, _generation_mode
    
    # 检测是否为生成模式 (seq_length == 1 通常表示生成新token)
    is_generating = seq_length == 1 and layer_id in _layer_routing_cache

    if is_generating:
        snapshot_routing_cache('/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/llama-factory/eval/routing_visual/gsm', "before_generate")
        # 生成模式：累积routing_weights
        cached_weights = _layer_routing_cache[layer_id]
        
        # 将新token的routing_weights追加到历史中
        full_routing_weights = torch.cat([cached_weights, routing_weights.view(batch_size, 1, -1)], dim=1) # B, S+1, E
        seq_length = full_routing_weights.size(1)  # 更新序列长度

        # 更新缓存
        _layer_routing_cache[layer_id] = full_routing_weights.detach()
        
        _, top1_indices = torch.topk(full_routing_weights, k=1, dim=-1)

        _, flat_indices = torch.topk(full_routing_weights.view(batch_size, -1), k=top_k * seq_length, dim=1)
        
        select_mask = torch.zeros_like(full_routing_weights.view(batch_size, -1), dtype=torch.bool).scatter_(-1, flat_indices, True).reshape(full_routing_weights.shape)
        select_mask.scatter_(-1, top1_indices, True)

        
        # 只对最后一个token（新生成的）应用mask
        last_token_mask = select_mask[:, -1, :]  # (batch_size, num_experts)
        last_token_routing = routing_weights.view(batch_size, 1, -1)[:, 0, :]  # (batch_size, num_experts)
        filtered_scores = (last_token_routing * last_token_mask.detach()).view(-1, moe_block.num_experts)
        expert_counts = (last_token_mask > 0).sum(dim=-1) 
        max_experts_per_token = expert_counts.max()
        routing_weights_topk, selected_experts = torch.topk(
            filtered_scores, max_experts_per_token, dim=-1
        )
    
    else:
        # 非生成模式或首次调用：直接执行sample_topk并初始化缓存
        routing_weights_reshaped = routing_weights.view(batch_size, seq_length, -1)  # (N, Seq_length, Expert)
        _, top1_indices = torch.topk(routing_weights_reshaped, k=1, dim=-1)

        _, flat_indices = torch.topk(routing_weights_reshaped.view(batch_size, -1), k=top_k * seq_length, dim=1)
        
        select_mask = torch.zeros_like(routing_weights_reshaped.view(batch_size, -1), dtype=torch.bool).scatter_(-1, flat_indices, True).reshape(routing_weights_reshaped.shape)
        select_mask.scatter_(-1, top1_indices, True)
        expert_counts = (select_mask > 0).sum(dim=-1) 
        max_experts_per_token = expert_counts.max()
        filtered_scores = (routing_weights_reshaped * select_mask.detach()).view(-1, moe_block.num_experts)
        routing_weights_topk, selected_experts = torch.topk(filtered_scores, k=max_experts_per_token, dim=1)
        
        # 初始化缓存
        _layer_routing_cache[layer_id] = routing_weights_reshaped.detach()

    return routing_weights_topk, selected_experts


def clear_routing_cache():
    """清空routing缓存 - 在每次新的生成开始时调用"""
    global _layer_routing_cache
    _layer_routing_cache.clear()


def set_generation_mode(enabled: bool):
    """设置生成模式"""
    global _generation_mode
    _generation_mode = enabled

def get_routing_cache_copy():
    # 返回一个可安全使用的拷贝（避免原地修改）
    return {k: v.detach().clone() for k, v in _layer_routing_cache.items()}

def save_routing_cache(save_dir: str, tag: str = "snapshot"):
    """
    将当前所有层的 routing_weights 缓存保存到磁盘。
    保存文件：{save_dir}/routing_{tag}.pt
    内容：{str(layer_id): Tensor[B, S, E] (CPU)}
    """
    os.makedirs(save_dir, exist_ok=True)
    payload = get_routing_cache_copy()
    torch.save(payload, os.path.join(save_dir, f"routing_{tag}.pt"))

def snapshot_routing_cache(save_dir: str, when: str):
    # alias，便于上层调用时语义化
    print('caching routing weights at {}'.format(when))
    save_routing_cache(save_dir, when)