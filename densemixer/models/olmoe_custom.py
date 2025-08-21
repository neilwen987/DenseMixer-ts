import torch
import torch.nn.functional as F
from wandb import config
from ..logging_utils import log_custom_forward_usage
from .. import config as densemixer_config
_layer_routing_cache = {}
_generation_mode = False  # 标记是否在生成模式
class CustomOlmoeSparseMoeBlock:
    @staticmethod
    def forward(self, hidden_states: torch.Tensor):
        """
        forward_partscale_fixep_norm_dtch
        """
        log_custom_forward_usage("OLMoE")
        batch_size, seq_length, hidden_dim = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        flat_hidden = hidden_states.view(-1, hidden_dim)  # (B*seq_len, hidden_dim)
        N_tokens = flat_hidden.size(0)
        layer_id = getattr(self, 'layer_idx', id(self))
        if not hasattr(self, 'i'):
            self.i = 0
        self.i += 1

        # override topk
        override_topk = densemixer_config.topk
        if override_topk is not None:
            self.top_k = override_topk

        # Compute routing logic
        router_logits = self.gate(flat_hidden).to(dtype=dtype)  # (B*L, num_experts)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # (B*L, num_experts)
        if densemixer_config.topk_mode == "topk":
            print('using topk, topk: {}'.format(self.top_k))
            # Select top-k experts
            routing_weights_topk, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            # print('using topk')
        elif densemixer_config.topk_mode == "batch_topk":
            # Select top-k experts per batch.
            # print('using btopk  ')
            top_k_logits_flat, top_k_indices_flat = torch.topk(routing_weights.flatten(), self.top_k * N_tokens, dim=-1)
            routing_weights_btopk = torch.zeros_like(routing_weights.flatten()).scatter_(-1, top_k_indices_flat, top_k_logits_flat).reshape(routing_weights.shape)
            num_selected_per_token = (routing_weights_btopk > 0).sum(dim=-1)  # (N_tokens,)
            max_selected = int(num_selected_per_token.max().item())
            routing_weights_topk, selected_experts = torch.topk(routing_weights_btopk, max_selected, dim=-1)
        
        elif densemixer_config.topk_mode == "sample_topk":
            # print('using sample_topk')
            if self.training:
                # print('use traing')
                routing_weights_reshaped = routing_weights.view(batch_size, seq_length, -1)  # (N, Seq_length, Expert)
            
                _, top6_indices = torch.topk(routing_weights_reshaped, k=(self.topk + 2), dim=-1)
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
            else:
                # print('Using SPTopk,this is the {}-th call of this layer'.format(self.i))
                routing_weights_topk, selected_experts = handle_sample_topk_with_cache(
                    self, routing_weights, self.top_k, batch_size, seq_length, layer_id, N_tokens
                )
        if self.norm_topk_prob:
            norm_ratio = routing_weights_topk.sum(dim=-1, keepdim=True)
            # Normalize top-k routing weights
            routing_weights_topk = routing_weights_topk / norm_ratio
            # Only scale the selected top-k positions in routing_weights
            mask = F.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=1).to(dtype)
            # ------------------------------------Choose Section-----------------------------------------------
            # current --> partscale_fix_expert implementation
            routing_weights = routing_weights * (1.0 - mask) / norm_ratio.detach() + routing_weights * mask / norm_ratio

            # should be --> the gated implemenation, by comment out the line above and uncomment the two lines below
            # gated = routing_weights.detach() * mask + (routing_weights - routing_weights.detach())
            # routing_weights = gated / gated.sum(dim=-1, keepdim=True)
            # ------------------------------------Choose Section-----------------------------------------------

        routing_weights_topk = routing_weights_topk.to(dtype=dtype)

        # Convert full routing_weights to consistent dtype for dense accumulation
        routing_weights = routing_weights.to(dtype=dtype)

        # Prepare accumulators: one for dense_outputs, one for sparse_outputs
        dense_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)
        sparse_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)

        # For mapping top-k positions when accumulating sparse_outputs
        # selected_experts: (N_tokens, top_k)

        # TODO: calculate relevant output in inference
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # Compute current expert output for all tokens
            expert_output = expert_layer(flat_hidden).to(dtype=dtype)  # (N_tokens, hidden_dim)
            activation_mask = (selected_experts == expert_idx).any(dim=1).float().unsqueeze(-1).to(dtype)
            if expert_output.requires_grad:
                expert_output.register_hook(lambda grad, mask=activation_mask: grad * mask)
            expert_output = expert_output.to(dtype=dtype)
            # Dense accumulation: multiply by full routing weight and add
            weight_full = routing_weights[:, expert_idx].unsqueeze(-1)  # (N_tokens, 1)
            dense_outputs = dense_outputs + expert_output * weight_full

            # Sparse accumulation: find tokens where this expert is among top_k
            # matches: Boolean mask where selected_experts == expert_idx → shape (N_tokens, top_k)
            matches = (selected_experts == expert_idx)
            if matches.any():
                # locations: tuple of (token_indices, k_indices)
                token_indices, k_indices = torch.where(matches)
                # corresponding top-k weights
                weights_topk = routing_weights_topk[token_indices, k_indices].unsqueeze(-1)  # (num_matches, 1)
                # Accumulate sparse_outputs only for matched tokens
                sparse_outputs[token_indices] = sparse_outputs[token_indices] + expert_output[token_indices] * weights_topk

        # Combine sparse forward output and dense backward output
        if self.training:
            final_flat = sparse_outputs.detach() + (dense_outputs - dense_outputs.detach())
        else:
            final_flat = sparse_outputs.detach()
        final_flat = final_flat.to(dtype=dtype)
        final_output = final_flat.view(batch_size, seq_length, hidden_dim)

        return final_output, router_logits

def handle_sample_topk_with_cache(moe_block, routing_weights, top_k, batch_size, seq_length, layer_id, N_tokens):
    """处理带缓存的sample_topk"""
    global _layer_routing_cache, _generation_mode
    
    # 检测是否为生成模式 (seq_length == 1 通常表示生成新token)
    is_generating = seq_length == 1 and layer_id in _layer_routing_cache

    if is_generating:
        # 生成模式：累积routing_weights
        cached_weights = _layer_routing_cache[layer_id]
        
        # 将新token的routing_weights追加到历史中
        full_routing_weights = torch.cat([cached_weights, routing_weights.view(batch_size, 1, -1)], dim=1) # B, S+1, E
        seq_length = full_routing_weights.size(1)  # 更新序列长度        
        _, top6_indices = torch.topk(full_routing_weights, k=(topk + 2), dim=-1)
        max_mask = torch.zeros_like(full_routing_weights, dtype=torch.bool).scatter_(-1, top6_indices, True)
        full_routing_weights = full_routing_weights * max_mask.detach()
            
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
        # 更新缓存  
        _layer_routing_cache[layer_id] = torch.cat([cached_weights, filtered_scores.view(batch_size, 1, -1)], dim=1)
    else:
        # 非生成模式或首次调用：直接执行sample_topk并初始化缓存
        routing_weights_reshaped = routing_weights.view(batch_size, seq_length, -1)  # (N, Seq_length, Expert)
        _, top6_indices = torch.topk(routing_weights_reshaped, k=(topk + 2), dim=-1)
        max_mask = torch.zeros_like(routing_weights_reshaped, dtype=torch.bool).scatter_(-1, top6_indices, True)
        routing_weights_reshaped = routing_weights_reshaped * max_mask.detach()
       
        _, top1_indices = torch.topk(routing_weights_reshaped, k=1, dim=-1)

        _, flat_indices = torch.topk(routing_weights_reshaped.view(batch_size, -1), k=top_k * seq_length, dim=1)
        
        select_mask = torch.zeros_like(routing_weights_reshaped.view(batch_size, -1), dtype=torch.bool).scatter_(-1, flat_indices, True).reshape(routing_weights_reshaped.shape)
        select_mask.scatter_(-1, top1_indices, True)
        expert_counts = (select_mask > 0).sum(dim=-1) 
        max_experts_per_token = expert_counts.max()
        filtered_scores = (routing_weights_reshaped * select_mask.detach()).view(-1, moe_block.num_experts)
        routing_weights_topk, selected_experts = torch.topk(filtered_scores, k=max_experts_per_token, dim=1)
        
        # 初始化缓存
        _layer_routing_cache[layer_id] = (routing_weights_reshaped * select_mask).detach()

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