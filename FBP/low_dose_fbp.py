import cv2
import numpy as np
from skimage.transform import radon, iradon
from skimage.io import imsave

# ==== 参数 ====
input_path = "/home/phoenix1943/桌面/fake/img_2.jpg"         # 输入 Ground Truth 图像路径
output_path = "recon_2.png"        # 输出重建图像路径
img_size = 512                   # 图像尺寸（会resize到这个大小）
num_angles = 720                 # 模拟全剂量投影角度数
sparse_ratio = 8                 # 稀疏比例（1/8剂量）

# ==== 图像预处理 ====
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (img_size, img_size))
img = img.astype(np.float32) / 255.0  # 归一化到 [0,1]

# ==== Radon 投影（模拟 1/8 剂量）====
full_angles = np.linspace(0., 180., num_angles, endpoint=False)
sparse_angles = full_angles[::sparse_ratio]
sinogram = radon(img, theta=sparse_angles, circle=True)

# ==== FBP 重建 ====
recon = iradon(sinogram, theta=sparse_angles, circle=True, filter_name='ramp')
recon = np.clip(recon, 0, 1)

# ==== 保存结果 ====
imsave(output_path, (recon * 255).astype(np.uint8))
print(f"重建完成：已保存到 {output_path}")
