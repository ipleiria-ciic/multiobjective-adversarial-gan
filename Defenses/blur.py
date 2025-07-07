import os
import torch
import utils
from scipy.ndimage import gaussian_filter
from torchvision.utils import save_image

dataroot = "../Dataset/Imagewoof"
save_root = "Blur_G"
batch_size = 128

os.makedirs(save_root, exist_ok=True)
device = utils.use_device()
dataloader = utils.get_dataloader(dataroot, image_size=128, batch_size=batch_size, workers=4)

index = 0
for i, (real_images, _) in enumerate(dataloader):
    real_images = real_images.to(device)
    blurred_images = torch.stack([
        torch.tensor(gaussian_filter(img.cpu().numpy(), sigma=1)) for img in real_images
    ]).to(device)

    for j, img in enumerate(blurred_images):
        print(f"[ INFO ] Saving image {index:06d}")
        save_path = os.path.join(save_root, f"img_{index:06d}.png")
        save_image(img, save_path)
        index += 1
