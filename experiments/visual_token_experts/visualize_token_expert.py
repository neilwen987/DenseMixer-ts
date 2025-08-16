import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 1. 数据加载 (请您使用您自己的数据加载方式替换掉这里的模拟数据)
# --------------------------------------------------------------------
# 模拟数据生成，为了让代码可以独立运行
def create_mock_data(n=1, num_tokens=20):
    mock_data = []
    tokens = ["This", "is", "a", "more", "beautiful", "visualization", "with", "rounded",
              "corners", "a", "title", "and", "a", "colorbar", "to", "show", "the",
              "scale", "of", "expert", "counts", ".", "Higher", "values", "are", "darker", "."]
    for i in range(n):
        np.random.seed(i)
        avg_exp_nums = np.random.rand(len(tokens)) * 8
        entry = {
            'step': i,
            'prompt_len': 0,
            'token_ids': [0] * len(tokens),
            'tokens': tokens,
            'routing_last': None,
            'avg_exp_num': torch.tensor(avg_exp_nums, dtype=torch.float32),
            'token_entropy': None
        }
        mock_data.append(entry)
    return mock_data

# 使用模拟数据
# data = create_mock_data()

# == 请您使用下面这行代码来加载您的真实数据 ==
data = torch.load('/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/visual_token_experts/routing_weights/generation_log_6.pt')
# --------------------------------------------------------------------
del data[-1]
del data[-1]

# 2. 数据提取与处理
if data and isinstance(data, list) and len(data) > 0:
    all_tokens = []
    all_avg_exp_num_list = []

    # 遍历Tuple中的每一个字典
    for item in data:
        # 确保 'tokens' 和 'avg_exp_num' 键存在于字典中
        if 'tokens' in item and 'avg_exp_num' in item:
            all_tokens.extend(item['tokens'])
            all_avg_exp_num_list.append(item['avg_exp_num'])
        else:
            print(f"警告：在Tuple的一个元素中缺少 'tokens' 或 'avg_exp_num' 键。")

    # 检查是否成功提取了数据
    if not all_tokens:
        print("数据Tuple中未找到有效的Token或avg_exp_num数据。")
        exit()

    # 将所有数据连接成一个大的列表/数组
    tokens = all_tokens
    avg_exp_num = np.array(all_avg_exp_num_list)

    # 对所有token的avg_exp_num进行统一的规范化
    min_val = avg_exp_num.min()
    max_val = avg_exp_num.max()
    normalized_values = (avg_exp_num - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(avg_exp_num)

else:
    print("数据为空或格式不正确（应为非空Tuple），无法生成图片。")
    exit()

# 3. 设置美化参数
# -- 颜色 --
BG_COLOR = '#DDDDDD' # 背景色-浅灰色
TEXT_COLOR_DARK = (0, 0, 0) # 深色文字
TEXT_COLOR_LIGHT = (0, 0, 0) # 浅色文字
COLOR_LIGHT = np.array([153, 255, 153]) # 渐变浅色-浅蓝
COLOR_DARK = np.array([0, 102, 255])    # 渐变深色-深蓝

# -- 布局 --
PADDING = 30
TITLE_HEIGHT = 0
COLORBAR_WIDTH = 80
TEXT_AREA_WIDTH = 700
TOTAL_WIDTH = TEXT_AREA_WIDTH + COLORBAR_WIDTH + PADDING

# -- 字体 --
FONT_SIZE_TITLE = 24
FONT_SIZE_TEXT = 18
FONT_SIZE_LABEL = 14
try:
    font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", FONT_SIZE_TITLE)
    font_text = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE_TEXT)
    font_label = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE_LABEL)
except IOError:
    print("未找到 DejaVuSans 字体, 使用默认字体。")
    font_title = ImageFont.load_default()
    font_text = ImageFont.load_default()
    font_label = ImageFont.load_default()

# -- 元素样式 --
TOKEN_PADDING = 5
TOKEN_SPACING = 4
LINE_SPACING = 12
CORNER_RADIUS = 8

# 4. 动态计算图片高度
def calculate_image_height():
    # 使用一个虚拟的Draw对象进行计算
    temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    x, y = PADDING, PADDING + TITLE_HEIGHT
    max_height = 0
    for token in tokens:
        bbox = temp_draw.textbbox((0, 0), token, font=font_text)
        token_width = bbox[2] - bbox[0]
        token_height = bbox[3] - bbox[1]

        if x + token_width + TOKEN_PADDING * 2 > TEXT_AREA_WIDTH - PADDING:
            x = PADDING
            y += token_height + LINE_SPACING

        x += token_width + TOKEN_PADDING * 2 + TOKEN_SPACING
        max_height = y + token_height + LINE_SPACING

    return max_height + PADDING

image_height = calculate_image_height()

# 5. 创建画布并绘制
img = Image.new('RGB', (TOTAL_WIDTH, image_height), color=BG_COLOR)
d = ImageDraw.Draw(img)


# -- 绘制带颜色的Token --
current_x, current_y = PADDING, PADDING + TITLE_HEIGHT
for token, norm_val in zip(tokens, normalized_values):
    bg_color_array = COLOR_LIGHT + (COLOR_DARK - COLOR_LIGHT) * norm_val
    bg_color = tuple(bg_color_array.astype(int))
    text_color = TEXT_COLOR_LIGHT if norm_val > 0.6 else TEXT_COLOR_DARK

    bbox = d.textbbox((0, 0), token, font=font_text)
    token_width = bbox[2] - bbox[0]
    token_height = bbox[3] - bbox[1]

    if current_x + token_width + TOKEN_PADDING * 2 > TEXT_AREA_WIDTH - PADDING:
        current_x = PADDING
        current_y += token_height + LINE_SPACING

    rect_start = (current_x, current_y)
    rect_end = (current_x + token_width + TOKEN_PADDING * 2, current_y + token_height + TOKEN_PADDING)
    d.rounded_rectangle([rect_start, rect_end], radius=CORNER_RADIUS, fill=bg_color)

    text_position = (current_x + TOKEN_PADDING, current_y + TOKEN_PADDING / 2)
    d.text(text_position, token, fill=text_color, font=font_text)

    current_x += token_width + TOKEN_PADDING * 2 + TOKEN_SPACING

# -- 绘制颜色条 (Color Bar) --
bar_x = TEXT_AREA_WIDTH
bar_y_start = PADDING + TITLE_HEIGHT
bar_height = image_height - (PADDING + TITLE_HEIGHT) - PADDING
bar_width = 20

# 绘制渐变矩形
for i in range(bar_height):
    norm_val = 1 - (i / bar_height) # 从上到下，值从大到小
    color_array = COLOR_LIGHT + (COLOR_DARK - COLOR_LIGHT) * norm_val
    bar_color = tuple(color_array.astype(int))
    d.line([(bar_x, bar_y_start + i), (bar_x + bar_width, bar_y_start + i)], fill=bar_color)

# 绘制颜色条边框
d.rectangle([(bar_x, bar_y_start), (bar_x + bar_width, bar_y_start + bar_height)], outline=(150, 150, 150))

# 绘制颜色条标题和标签
bar_title = "Avg. Exp Num"
bar_title_bbox = d.textbbox((0,0), bar_title, font=font_label)
bar_title_width = bar_title_bbox[2] - bar_title_bbox[0]
d.text((bar_x + (bar_width - bar_title_width) / 2, bar_y_start - 20), bar_title, fill=TEXT_COLOR_DARK, font=font_label)

max_val_str = f"{max_val:.2f}"
min_val_str = f"{min_val:.2f}"

d.text((bar_x + bar_width + 5, bar_y_start - 5), max_val_str, fill=TEXT_COLOR_DARK, font=font_label)
d.text((bar_x + bar_width + 5, bar_y_start + bar_height - 10), min_val_str, fill=TEXT_COLOR_DARK, font=font_label)


import os
# 6. 保存图片
path = '/home/ubuntu/tiansheng/26_ICLR_btk_moe/DenseMixer-ts/experiments/visual_token_experts/fig'
output_path = os.path.join(path,'token_visualization.png')
img.save(output_path)

print(f"图片已成功生成并保存至: {output_path}")