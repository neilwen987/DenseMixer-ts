import torch
import torch.nn.functional as F
from wandb import config
from ..logging_utils import log_custom_forward_usage
from .. import config as densemixer_config
class ConventionalOlmoeSparseMoeBlock:
    @staticmethod
    def forward(self, hidden_states: torch.Tensor):
        """
        forward_partscale_fixep_norm_dtch
        """
        log_custom_forward_usage("OLMoE")
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        if densemixer_config.topk_mode == "topk":
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        elif densemixer_config.topk_mode == "sample_topk":
            top_k_logits_flat, top_k_indices_flat = torch.topk(routing_weights.view(batch_size,-1), self.top_k * sequence_length, dim=-1)
            routing_weights_btopk = torch.zeros_like(routing_weights.view(batch_size,-1)).scatter_(-1, top_k_indices_flat, top_k_logits_flat).reshape(routing_weights.shape)
            num_selected_per_token = (routing_weights_btopk > 0).sum(dim=-1)  # (N_tokens,)
            max_selected = int(num_selected_per_token.max().item())
            routing_weights, selected_experts = torch.topk(routing_weights_btopk, max_selected, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits