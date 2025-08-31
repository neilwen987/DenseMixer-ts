from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from .. import config as densemixer_config
from ..logging_utils import log_custom_forward_usage

class CustomGptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.alpha = 1.702
        self.limit = 7.0

        # Initialize expert parameters to avoid NaNs from uninitialized memory
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for expert_idx in range(self.num_experts):
            nn.init.xavier_uniform_(self.gate_up_proj[expert_idx])
            nn.init.zeros_(self.gate_up_proj_bias[expert_idx])
            nn.init.xavier_uniform_(self.down_proj[expert_idx])
            nn.init.zeros_(self.down_proj_bias[expert_idx])

    def forward_ds(self, hidden_states: torch.Tensor, router_indices=None, routing_weights_topk=None, routing_weights_full=None) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
        Returns:
            torch.Tensor
        """
        # Shapes
        if hidden_states.dim() == 3:
            batch_size, seq_len, _ = hidden_states.shape
        else:
            batch_size, seq_len = 1, hidden_states.shape[0]

        flat_states = hidden_states.reshape(-1, self.hidden_size)  # (N_tokens, hidden_size)
        num_tokens = flat_states.size(0)
        num_experts = self.num_experts

        # Ensure routing tensors are provided
        assert routing_weights_topk is not None and routing_weights_full is not None, "Both topk and full routing weights are required"

        # Compute all experts' outputs for all tokens using batched matmuls
        hs_batched = flat_states.repeat(num_experts, 1).view(num_experts, -1, self.hidden_size)  # (E, N, H)
        gate_up = torch.bmm(hs_batched, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]  # (E, N, 2D)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        expert_out = torch.bmm(((up + 1) * glu), self.down_proj)  # (E, N, H)
        expert_out = expert_out + self.down_proj_bias[..., None, :]

        # Gradient masking so experts only receive token-level grads when selected
        if expert_out.requires_grad and router_indices is not None:
            # router_indices: (N_tokens, top_k)
            for expert_idx in range(num_experts):
                with torch.no_grad():
                    active_mask = (router_indices == expert_idx).any(dim=1).float().unsqueeze(-1)  # (N, 1)
                expert_out[expert_idx].register_hook(lambda grad, mask=active_mask: grad * mask)

        # Prepare weights
        weights_full = routing_weights_full.to(dtype=expert_out.dtype)  # (N, E)
        weights_sparse = routing_weights_topk.to(dtype=expert_out.dtype)  # (N, E) zero outside top-k

        # Dense accumulation over all experts (for backward path)
        dense_outputs = (expert_out * weights_full.t().unsqueeze(-1)).sum(dim=0)  # (N, H)

        # Sparse accumulation using only top-k weights (for forward path)
        sparse_outputs = (expert_out * weights_sparse.t().unsqueeze(-1)).sum(dim=0)  # (N, H)

        # Straight-through style combine: sparse forward, dense backward
        if self.training:
            final_flat = sparse_outputs.detach() + (dense_outputs - dense_outputs.detach())
        else:
            final_flat = sparse_outputs.detach()

        next_states = final_flat.view(batch_size, seq_len, self.hidden_size)
        return next_states
    
    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]
        if hidden_states.device.type == "cpu" or self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                # we sum on the top_k and on the sequence length to get which experts
                # are hit this time around
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit[:]:
                # expert_idx only have 1 element, so we can use scale for fast indexing
                expert_idx = expert_idx[0]
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
                weighted_output = out * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.gate_up_proj_bias[..., None, :]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            next_states = torch.bmm(((up + 1) * glu), self.down_proj)
            next_states = next_states + self.down_proj_bias[..., None, :]
            next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
            next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            next_states = next_states.sum(dim=0)
        return next_states

class CustomGptOssTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

    def forward(self, hidden_states):
        # 将输入展平以计算路由 logits
        if hidden_states.dim() == 3:
            batch_size, seq_length, _ = hidden_states.shape
        else:
            batch_size, seq_length = 1, hidden_states.shape[0]
        flat_states = hidden_states.reshape(-1, self.hidden_dim)
        # override topk if configured
        override_topk = densemixer_config.topk
        if override_topk is not None:
            self.top_k = override_topk

        router_logits = F.linear(flat_states, self.weight, self.bias)  # (N_tokens, num_experts)
        # full softmax 权重（用于后续 dense backward）
        routing_weights_full = F.softmax(router_logits, dim=1, dtype=torch.float)
        # USE pre-softmax for consistency with dense backward

        if densemixer_config.topk_mode == "topk":
            # 先 topk 再在选中集合上 softmax
            router_top_value, router_indices = torch.topk(routing_weights_full, self.top_k, dim=-1)
            router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        
        elif densemixer_config.topk_mode == "sample_topk":
            # pre-softmax 概率
            probs_bsxe = routing_weights_full.view(batch_size, seq_length, self.num_experts)

            cap_topk = densemixer_config.cap_topk
            if cap_topk is not None:
                cap_k = max(1, int(cap_topk))
                # 逐 token 截断：每个 token 仅保留其前 cap_k 个 expert（按概率）
                _, cap_indices_tok = torch.topk(probs_bsxe, k=cap_k, dim=-1)
                cap_mask = torch.zeros_like(probs_bsxe, dtype=torch.bool).scatter(-1, cap_indices_tok, True)
            else:
                cap_k = None
                cap_mask = torch.ones_like(probs_bsxe, dtype=torch.bool)

            # 每个 batch 选出若干 (token, expert) 位置（按概率）
            k_per_batch = max(1, (self.top_k - 1) * int(seq_length))
            probs_cap = probs_bsxe.masked_fill(~cap_mask, 0.0)
            _, flat_indices = torch.topk(probs_cap.view(batch_size, -1), k=k_per_batch, dim=1)
            select_mask = torch.zeros_like(probs_bsxe.view(batch_size, -1), dtype=torch.bool)
            select_mask = select_mask.scatter(-1, flat_indices, True).view_as(probs_bsxe)

            # 保障每个 token 至少包含其 top1 expert（按概率）
            _, top1_indices = torch.topk(probs_bsxe, k=1, dim=-1)
            select_mask = select_mask.scatter(-1, top1_indices, True)
            allowed_mask = (select_mask & cap_mask).scatter(-1, top1_indices, True)

            # 仅在允许集合上选出最终 K（按概率），并在选中集合内重新归一化
            masked_probs = probs_bsxe.masked_fill(~allowed_mask, 0.0).view(-1, self.num_experts)
            topk_values, router_indices = torch.topk(masked_probs, k=cap_k, dim=-1)
            selected_probs = torch.gather(masked_probs, 1, router_indices)
            norm = selected_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
            final_probs = selected_probs / norm
            router_scores = torch.zeros_like(router_logits)
            router_scores.scatter_(1, router_indices, final_probs)
        else:
            assert False, "Invalid topk mode"
        return router_scores, router_indices, routing_weights_full



class CustomGptOssMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = CustomGptOssTopKRouter(config)
        self.experts = CustomGptOssExperts(config)

    def forward(self, hidden_states):
        # Log once that we are using DenseMixer's custom forward
        log_custom_forward_usage("GPT-OSS")
        router_scores, router_indices, routing_weights_full = self.router(hidden_states)
        routed_out = self.experts(
            hidden_states,
            router_indices=router_indices,
            routing_weights_topk=router_scores,
            routing_weights_full=routing_weights_full,
        )
        return routed_out, router_scores