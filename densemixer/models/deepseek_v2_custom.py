import torch
import torch.nn.functional as F
from ..logging_utils import log_custom_forward_usage
from .. import config as densemixer_config

class CustomDeepseekV2MoE:
    @staticmethod
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # print('using custom dsv2')
        residuals = hidden_states
        orig_shape = hidden_states.shape
        batch_size, seq_length, hidden_dim = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        selected_experts,routing_weights_topk, routing_weights = self.gate(hidden_states)
        num_experts = routing_weights.shape[1]
        # print('num_experts: {}'.format(num_experts))
        # hidden_states = hidden_states.view(-1, hidden_states.shape[-1]) # (B*seq_len, hidden_dim)

        flat_hidden = hidden_states.view(-1, hidden_dim)  # (B*seq_len, hidden_dim)
        N_tokens = flat_hidden.size(0)
        layer_id = getattr(self, 'layer_idx', id(self))
        if not hasattr(self, 'i'):
            self.i = 0
        self.i += 1       

        routing_weights_topk = routing_weights_topk.to(dtype=dtype)
        routing_weights = routing_weights.to(dtype=dtype)
        shared_expert_output = self.shared_experts(residuals)

        dense_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)
        sparse_outputs = torch.zeros((N_tokens, hidden_dim), dtype=dtype, device=device)

        for expert_idx in range(num_experts):
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

        sparse_outputs = sparse_outputs.view(*orig_shape) + shared_expert_output
        dense_outputs = dense_outputs.view(*orig_shape) + shared_expert_output

        # Combine sparse forward output and dense backward output
        if self.training:
            final_flat = sparse_outputs.detach() + (dense_outputs - dense_outputs.detach())
        else:
            final_flat = sparse_outputs.detach()
        final_flat = final_flat.to(dtype=dtype)
        return final_flat


class CustomDeepseekV2MoEGate:
    @staticmethod

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape

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


        ### compute gating score
        hidden_states = hidden_states.view(-1, hidden_dim)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        # select top-k experts
        # greedy method is used for DeepSeek-V2-Lite
        # group_limited_greedy for DeepSeek-V2 and DeepSeek-V2-Chat
        if self.topk_method == "greedy":
            if densemixer_config.topk_mode == "topk":
                # print('using topk, topk: {}'.format(self.top_k))
                topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        elif self.topk_method == "group_limited_greedy":
            group_scores = scores.view(batch_size * seq_len, self.num_group, -1).max(dim=-1).values  # [n, num_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, num_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, num_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(batch_size * seq_len, self.num_group, self.num_experts // self.num_group)
                .reshape(batch_size * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)

        topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        return topk_idx, topk_weight, scores

