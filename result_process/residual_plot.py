import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

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

abs_residuals = [
    ("FBP", np.abs(ground_truth - fbp_reconstruction)),
    ("SIN-4c-PRN", np.abs(ground_truth - sin_4c_prn)),
    ("Score-SDE", np.abs(ground_truth - score_sde)),
    ("Proposed", np.abs(ground_truth - ours)),
    
]

colors = ["#253091", "#00FFFF", "#FFFF00", "#FFFFFF"]
custom_cmap = LinearSegmentedColormap.from_list("custom_bluehot", colors, N=256)

vmax = max([res.max() for _, res in abs_residuals])

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, (label, residual) in enumerate(abs_residuals):
    row, col = divmod(i, 2)
    im = axs[row, col].imshow(residual, cmap=custom_cmap, vmin=0, vmax=vmax)
    axs[row, col].set_title(f"{label}", fontsize=25)
    axs[row, col].axis('off')

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()
