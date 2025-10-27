import torch
from densemixer.models.gptoss_moe_custom import CustomGptOssTopKRouter
from densemixer.models.gptoss_moe_custom import CustomGptOssExperts
from types import SimpleNamespace
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter
import os
# os.environ['DENSEMIXER_ENABLED'] = '1'
# os.environ['DENSEMIXER_GPT_OSS'] = '1'
os.environ['DENSEMIXER_CAP_TOPK'] = '3'

config = SimpleNamespace(
    hidden_size=4,
    intermediate_size=8,
    num_local_experts=4,
    num_experts_per_tok=2,
)

router = GptOssTopKRouter(config)
experts = CustomGptOssExperts(config)

hidden_states = torch.randn(2, 8, 4)
router_scores, router_indices,  = router(hidden_states)
# print(routing_weights_full)
print(router_scores)
print(router_indices)

# output1 = experts.forward_ds(hidden_states, router_indices, router_scores, routing_weights_full)
# print(output1)

# output2 = experts.forward(hidden_states, router_indices, router_scores)
# print(output2)

