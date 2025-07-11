import os
import argparse
import torch
import utils
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--blur", type=str, choices=["gaussian", "median", "bilateral", "affine"])
args = parser.parse_args()

blur_type = args.blur

# Setup
dataroot = "Dataset/Imagewoof"
save_root = f"Blurred_Images_{blur_type.capitalize()}"
batch_size = 128

os.makedirs(save_root, exist_ok=True)
device = utils.use_device()
dataloader = utils.get_dataloader(dataroot, image_size=128, batch_size=batch_size, workers=4)

index = 0
for i, (real_images, _, paths) in enumerate(dataloader):
    real_images = real_images.to(device)

    blurred_images = []
    for img in real_images:
        img_np = img.cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np * 255).astype(np.uint8)

        if blur_type == "gaussian":
            blurred = gaussian_filter(img_np, sigma=1)
        elif blur_type == "median":
            blurred = median_filter(img_np, size=3)
        elif blur_type == "bilateral":
            blurred = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
        elif blur_type == "affine":
            rows, cols, _ = img_np.shape
            M = np.float32([[1, 0.2, 0], [0.2, 1, 0]])
            blurred = cv2.warpAffine(img_np, M, (cols, rows))

        blurred = blurred.astype(np.float32) / 255.0
        blurred = np.transpose(blurred, (2, 0, 1))
        blurred_images.append(torch.tensor(blurred))

    blurred_images = torch.stack(blurred_images).to(device)

    for j, img in enumerate(blurred_images):
        original_path = paths[j]
        class_name = os.path.basename(os.path.dirname(original_path))
        filename = os.path.basename(original_path)
        
        class_dir = os.path.join(save_root, class_name)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, filename)
        print(f"[ INFO ] Saving image to {save_path}")
        save_image(img, save_path)