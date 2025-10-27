import numpy as np
import matplotlib.pyplot as plt

# 读取并展平
entropy = np.load("/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/visual_token_experts/routing_weights/entropy.npy").reshape(-1)
exp_num = np.load("/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/visual_token_experts/routing_weights/exp_num.npy").reshape(-1)

# 定义分桶
bins = [0, 0.5, 1, 2,3,7]
bin_labels = ["0-0.5", "0.5-1", "1-2",'2-3','>3']

avg_exp_per_bin = []

for i in range(len(bins) - 1):
    mask = (entropy >= bins[i]) & (entropy < bins[i+1])
    if np.any(mask):
        avg_exp_per_bin.append(np.mean(exp_num[mask]))
    else:
        avg_exp_per_bin.append(0)

plt.figure(figsize=(8, 6))
plt.bar(bin_labels, avg_exp_per_bin, color='skyblue', edgecolor='k')
plt.xlabel("Entropy Range", fontsize=14)
plt.ylabel("Average Exp Num", fontsize=14)
plt.title("Average Exp Num by Entropy Range", fontsize=16)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/visual_token_experts/routing_weights/entropy_expnum_hist.png", dpi=300)  # 保存图片
plt.show()