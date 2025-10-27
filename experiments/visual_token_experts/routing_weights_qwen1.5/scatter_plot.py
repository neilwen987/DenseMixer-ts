import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import torch
import os
# --- 1. 加载数据 ---
from datasets import load_dataset, get_dataset_config_names
dataset_name = "EleutherAI/hendrycks_math"
subset_names = ['algebra','counting_and_probability','geometry','intermediate_algebra',
                'number_theory','precalculus']

root_path = '/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/visual_token_experts/routing_weights'
name = 'generation_metrics_all_batches_1.pt'

list_exp =[]
list_entorpy=[]
for tasks in subset_names:
    print(tasks)
    data = torch.load(os.path.join(root_path,tasks,name))
    list_exp.extend(data['avg_exp_num'])
    list_entorpy.extend(data['token_entropy'])

print('len exp:', len(list_entorpy))
np_exp = np.array(list_exp)
np_entropy = np.array(list_entorpy)
# import pdb
# pdb.set_trace()
try:
    entropy = np.load(os.path.join(root_path,"entropy.npy")).reshape(-1)
    exp_num = np.load(os.path.join(root_path,"exp_num.npy")).reshape(-1)

    entropy = np.concatenate([np_entropy, entropy], axis=0)
    exp_num = np.concatenate([np_exp, exp_num], axis=0)

    mask = entropy > 8
    entropy = entropy[mask]
    exp_num = 

    # --- 2. 绘制散点图 + 回归线 ---
    plt.figure(figsize=(10, 6))
    plt.scatter(entropy, exp_num, alpha=0.5, label="Data points")

    # 计算回归线参数
    slope, intercept = np.polyfit(entropy, exp_num, 1)
    x_vals = np.linspace(entropy.min(), entropy.max(), 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color="red", linewidth=2, label=f"y = {slope:.3f}x + {intercept:.3f}")

    plt.title('Scatter Plot of Entropy vs. Expert Number with Regression Line')
    plt.xlabel('Entropy')
    plt.ylabel('Expert Number')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/visual_token_experts/routing_weights/scatter_with_regression_new.png", dpi=300)
    plt.show()

    # --- 3. 相关性检验 ---
    corr, p_value = pearsonr(entropy, exp_num)
    print(f"皮尔逊相关系数 (Corr): {corr}")
    print(f"P值 (P-value): {p_value}")

    # --- 4. 结果解读 ---
    corr_desc = "正相关" if corr > 0 else "负相关" if corr < 0 else "不相关"
    print(f"结果表明存在 {corr_desc}。")
    if p_value < 0.05:
        print("在0.05的显著性水平上，entropy和exp_num之间存在显著的相关性。")
    else:
        print("在0.05的显著性水平上，entropy和exp_num之间没有显著的相关性。")

except FileNotFoundError:
    print("错误：一个或两个npy文件未找到。请检查文件路径是否正确。")
except Exception as e:
    print(f"发生错误: {e}")