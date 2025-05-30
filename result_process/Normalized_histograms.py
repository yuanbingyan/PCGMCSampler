import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

mask = Image.new('L', (512, 512), 0)
draw = ImageDraw.Draw(mask)
draw.pieslice(
    [0, 0, 512, 512],
    0, 
    360,
    fill=255
)
mask = np.array(mask)/255

# Select image to calculate PSNR & SSIM

fbp_reconstruction = np.float32(
    cv2.imread(
        '/home/phoenix1943/桌面/FBP_reconstruction/recon_1.png', 0
    )
)
ours = np.float32(
    cv2.imread(
        '/home/phoenix1943/桌面/ours/blurred_1_0.5.png', 0
    )
)
score_sde = np.float32(
    cv2.imread(
        '/home/phoenix1943/桌面/score_sde/blurred_1_1.2.png', 0
    )
)
sin_4c_prn = np.float32(
    cv2.imread(
        '/home/phoenix1943/桌面/SIN-4c-PRN/blurred_1_1.5.png', 0
    )
)
ground_truth = np.float32(
    cv2.imread(
        '/home/phoenix1943/桌面/ground_truth/img_1.jpg', 0
    )
)

fbp_reconstruction /= 255.
ours /= 255.
score_sde /= 255.
sin_4c_prn /= 255.
ground_truth /= 255.

fbp_reconstruction = torch.tensor(fbp_reconstruction, dtype=torch.float32)
ours = torch.tensor(ours, dtype=torch.float32)
score_sde = torch.tensor(score_sde, dtype=torch.float32)
sin_4c_prn = torch.tensor(sin_4c_prn, dtype=torch.float32)
ground_truth = torch.tensor(ground_truth, dtype=torch.float32)

fbp_reconstruction_hist = torch.histc(fbp_reconstruction, bins=255)
ours_hist = torch.histc(ours, bins=255)
score_sde_hist = torch.histc(score_sde, bins=255)
sin_4c_prn_hist = torch.histc(sin_4c_prn, bins=255)
ground_truth_hist = torch.histc(ground_truth, bins=255)

threshold = 75
peak_fbp_reconstruction_1 = torch.argmax(fbp_reconstruction_hist[:threshold]) / 255.
peak_fbp_reconstruction_2 = (torch.argmax(fbp_reconstruction_hist[threshold:]) + threshold) / 255.

peak_ours_1 = torch.argmax(ours_hist[:threshold]) / 255.
peak_ours_2 = (torch.argmax(ours_hist[threshold:]) + threshold) / 255.

peak_score_sde_1 = torch.argmax(score_sde_hist[:threshold]) / 255.
peak_score_sde_2 = (torch.argmax(score_sde_hist[threshold:]) + threshold) / 255.

peak_sin_4c_prn_1 = torch.argmax(sin_4c_prn_hist[:threshold]) / 255.
peak_sin_4c_prn_2 = (torch.argmax(sin_4c_prn_hist[threshold:]) + threshold) / 255.

peak_ground_truth_1 = torch.argmax(ground_truth_hist[:threshold]) / 255.
peak_ground_truth_2 = (torch.argmax(ground_truth_hist[threshold:]) + threshold) / 255.

fbp_reconstruction = torch.clamp((fbp_reconstruction - peak_fbp_reconstruction_1) / (peak_fbp_reconstruction_2 - peak_fbp_reconstruction_1), min = 0)
ours = torch.clamp((ours - peak_ours_1) / (peak_ours_2 - peak_ours_1), min = 0)
score_sde = torch.clamp((score_sde - peak_score_sde_1) / (peak_score_sde_2 - peak_score_sde_1), min = 0)
sin_4c_prn = torch.clamp((sin_4c_prn - peak_sin_4c_prn_1) / (peak_sin_4c_prn_2 - peak_sin_4c_prn_1), min = 0)
ground_truth = torch.clamp((ground_truth - peak_ground_truth_1) / (peak_ground_truth_2 - peak_ground_truth_1), min = 0)

fbp_reconstruction = torch.clamp(fbp_reconstruction, max=torch.max(ground_truth), min=0)
ours = torch.clamp(ours, max=torch.max(ground_truth), min=0)
score_sde = torch.clamp(score_sde, max=torch.max(ground_truth), min=0)
sin_4c_prn = torch.clamp(sin_4c_prn, max=torch.max(ground_truth), min=0)

fbp_reconstruction /= torch.max(ground_truth)
ours /= torch.max(ground_truth)
score_sde /= torch.max(ground_truth)
sin_4c_prn /= torch.max(ground_truth)

ground_truth /= torch.max(ground_truth)

hist_fbp = np.histogram(np.array(fbp_reconstruction).ravel(), bins=255)[0]
hist_sin = np.histogram(np.array(sin_4c_prn).ravel(), bins=255)[0]
hist_score = np.histogram(np.array(score_sde).ravel(), bins=255)[0]
hist_ours = np.histogram(np.array(ours).ravel(), bins=255)[0]
hist_gt = np.histogram(np.array(ground_truth).ravel(), bins=255)[0]
y_max = max(hist_fbp.max(), hist_sin.max(), hist_score.max(), hist_ours.max(), hist_gt.max())

plt.figure(figsize=(10, 10))

plt.subplot(3, 2, 1)
plt.hist(np.array(fbp_reconstruction).ravel(), bins=255)
plt.ylim(0, y_max)
plt.xlim(0, 1)
plt.title('FBP', fontsize=20)

plt.subplot(3, 2, 2)
plt.hist(np.array(sin_4c_prn).ravel(), bins=255)
plt.ylim(0, y_max)
plt.xlim(0, 1)
plt.title('SIN-4c-PRN', fontsize=20)

plt.subplot(3, 2, 3)
plt.hist(np.array(score_sde).ravel(), bins=255)
plt.ylim(0, y_max)
plt.xlim(0, 1)
plt.title('Score-SDE', fontsize=20)

plt.subplot(3, 2, 4)
plt.hist(np.array(ours).ravel(), bins=255)
plt.ylim(0, y_max)
plt.xlim(0, 1)
plt.title('Proposed', fontsize=20)

plt.subplot(3, 2, 5)
plt.hist(np.array(ground_truth).ravel(), bins=255)
plt.ylim(0, y_max)
plt.xlim(0, 1)
plt.title('Ground truth', fontsize=20)

plt.show()
