import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# 加载预训练模型
model = keypointrcnn_resnet50_fpn(pretrained=True)

# 修改关键点数量（例如，杯子和笔各有 3 个关键点）
num_keypoints = 6  # 杯子 3 个 + 笔 3 个
model.roi_heads.keypoint_predictor.kps_score_lowres.out_channels = num_keypoints
model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels = 512

# 将模型切换到评估模式
model.eval()

from torchvision.transforms import functional as F
from PIL import Image

# 加载图像
image_path = "C:\\Users\\zy\\Desktop\\ReKep\\Rekep-main\\微信图片_20250227194623.jpg"
image = Image.open(image_path).convert("RGB")

# 转换为 Tensor
image_tensor = F.to_tensor(image).unsqueeze(0)  # 增加 batch 维度

# 模型推理
with torch.no_grad():
    outputs = model(image_tensor)

# 解析输出
for i, output in enumerate(outputs):
    keypoints = output["keypoints"]  # 关键点坐标
    scores = output["scores"]        # 置信度分数
    labels = output["labels"]        # 类别标签（杯子或笔）

    print(f"Object {i}:")
    for j, keypoint in enumerate(keypoints):
        print(f"Keypoint {j}: {keypoint}, Score: {scores[j]}, Label: {labels[j]}")

import cv2
import numpy as np

# 将图像转换为 NumPy 数组
image_np = np.array(image)

# 绘制关键点
for i, output in enumerate(outputs):
    keypoints = output["keypoints"].cpu().numpy()
    for keypoint in keypoints:
        x, y, _ = keypoint.astype(int)
        cv2.circle(image_np, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

# 显示结果
cv2.imshow("Keypoints", image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
