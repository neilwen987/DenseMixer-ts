import os
import torch
import numpy as np
from tqdm import tqdm

base_path = '/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/visual_token_experts/routing_weights'
entropy = []
exp_num = []

for file in tqdm(os.listdir(base_path)):
    if file.endswith('.pt'):
        file_path = os.path.join(base_path, file)
        data = torch.load(file_path, map_location="cpu")
        
        # 如果 data 是列表
        for item in data:
            entropy.append(item['token_entropy'])
            exp_num.append(item['avg_exp_num'])

print('len entropy:', len(entropy))
print('len avg_exp_num:', len(exp_num))

entropy = np.array(entropy)
exp_num = np.array(exp_num)

np.save(os.path.join(base_path, 'entropy.npy'), entropy)
np.save(os.path.join(base_path, 'exp_num.npy'), exp_num)