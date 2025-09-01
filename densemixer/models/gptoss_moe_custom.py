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
        # self._reset_parameters()

    # def _reset_parameters(self) -> None:
    #     for expert_idx in range(self.num_experts):
    #         nn.init.xavier_uniform_(self.gate_up_proj[expert_idx])
    #         nn.init.zeros_(self.gate_up_proj_bias[expert_idx])
    #         nn.init.xavier_uniform_(self.down_proj[expert_idx])
    #         nn.init.zeros_(self.down_proj_bias[expert_idx])

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

class CustomGptOssTopKRouter():
    @staticmethod
    def forward(self, hidden_states):
        batch_size, seq_length, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
        
        
        layer_id = getattr(self, 'layer_idx', id(self))
        if not hasattr(self, 'i'):
            self.i = 0
        self.i += 1

        # override topk
        override_topk = densemixer_config.topk
        if override_topk is not None:
            self.top_k = override_topk

        if densemixer_config.topk_mode == "topk":
            router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        elif densemixer_config.topk_mode == "sample_topk":
            if True:
                logits_bsxe = router_logits.view(batch_size, seq_length, self.num_experts)

                # adding cap
                cap_topk = densemixer_config.cap_topk =3
                if cap_topk is not None:
                    cap_k = max(1, int(cap_topk))
                    # 逐 token 截断：每个 token 仅保留其前 cap_k 个 expert
                    _, cap_indices_tok = torch.topk(logits_bsxe, k=cap_k, dim=-1)
                    cap_mask = torch.zeros_like(logits_bsxe, dtype=torch.bool).scatter(-1, cap_indices_tok, True)
                    logits_bsxe = logits_bsxe.masked_fill(~cap_mask, -float('inf'))
                else:
                    cap_k = self.top_k
                
                top1_logits, top1_indices = torch.topk(logits_bsxe, k=1, dim=-1)
                minimal_mask = torch.zeros_like(logits_bsxe, dtype=torch.bool).scatter(-1, top1_indices, True)
                logits_bsxe = logits_bsxe.masked_fill(minimal_mask, -float('inf'))

                flat_top_value, flat_top_indices = torch.topk(logits_bsxe.view(batch_size, -1), k=(self.top_k -1) * seq_length, dim=-1)
                final_score = torch.full_like(logits_bsxe.view(batch_size, -1), float('-inf'))
                final_score = final_score.scatter_(-1, flat_top_indices, flat_top_value).reshape(*logits_bsxe.shape)
                final_score = final_score.scatter_(-1, top1_indices, top1_logits).view(-1,self.num_experts)
                router_top_value, router_indices = torch.topk(final_score, cap_k, dim=-1)

            else:
                raise ValueError(f"inference not supported for sample_topk")   # (seq_len, top_k)
        elif densemixer_config.topk_mode == "batch_topk":
            router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        else:
            raise ValueError(f"Invalid topk mode: {densemixer_config.topk_mode}")
        
        
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices
    


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
            router_indices,
            router_scores,
        )
        return routed_out, router_scores