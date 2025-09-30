# import tiktoken
#
#
# def count_tokens_in_txt(file_path, model_name="gpt-4o"):
#     # 加载模型对应的编码器
#     encoding = tiktoken.encoding_for_model(model_name)
#
#     # 读取文本文件内容
#     with open(file_path, "r", encoding="utf-8") as f:
#         text = f.read()
#
#     # 计算token数
#     tokens = encoding.encode(text)
#     return len(tokens)
#
#
# # 示例：计算当前目录下prompt.txt的token数
# file_path = "C:\\Users\\zy\\Desktop\\ReKep\\Rekep-main\\vlm_query\\prompt_template.txt"
# token_count = count_tokens_in_txt(file_path)
# print(f"文件 '{file_path}' 包含 {token_count} 个token（模型：gpt-4o）。")

import tiktoken
import numpy as np
from PIL import Image

def count_image_tokens(image_path, model="gpt-4o"):
    # 加载图片并调整到最大支持尺寸（假设为512x512）
    img = Image.open(image_path)
    img = img.resize((512, 512))  # 缩放至512x512
    # 计算补丁数量（假设补丁大小为14x14）
    patches = (np.ceil(np.array(img.size) / 14)).astype(int)
    num_tokens = patches[0] * patches[1] + 2  # +2 为固定标记
    return num_tokens

# 示例：计算示例图片的Token数
token_count = count_image_tokens("C:\\Users\\zy\\Desktop\\ReKep\\Rekep-main\\vlm_query\\pen\\query_img.png")
print(f"图片包含 {token_count} 个Tokens")