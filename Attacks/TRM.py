import os
import torch
import argparse
import numpy as np
from alive_progress import alive_bar
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description="TRM Attack on Imagewoof Dataset")
parser.add_argument("--delta", type=str)
args = parser.parse_args()

output_dir = f"Attacks/TRM/TRM-{args.delta}"
imagewoof_path = "Dataset/Imagewoof/train"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=imagewoof_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

uap_path = f"Attacks/Perturbations/TRM.npy"
os.makedirs(os.path.dirname(uap_path), exist_ok=True)
uap = np.load(uap_path)
uap = torch.tensor(uap).to(device)

unnormalize = transforms.Compose([
    transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    ),
    transforms.ToPILImage(),
])

with alive_bar(len(dataloader), title=f"[ INFO ] Generating the TRM attack ({float(args.delta):.02f})", bar='classic', spinner=None) as bar:
    for i, (image_tensor, _) in enumerate(dataloader):
        image_tensor = image_tensor.to(device)
        uap = uap.mean(dim=0, keepdim=True)

        # Apply the universal perturbation
        perturbed_image_tensor = image_tensor + (uap * float(args.delta))

        perturbed_image_tensor = perturbed_image_tensor.squeeze(0)  # Remove extra batch dimension if needed

        # Convert back to image format for saving
        perturbed_image = unnormalize(perturbed_image_tensor)

        # Extract label from file path
        original_path, _ = dataset.samples[i]
        class_folder = os.path.basename(os.path.dirname(original_path))  # Get folder name

        class_output_folder = os.path.join(output_dir, class_folder)
        os.makedirs(class_output_folder, exist_ok=True)

        # Save perturbed image
        filename = os.path.basename(original_path)
        perturbed_image.save(os.path.join(class_output_folder, filename))

        bar()  # Update progress bar

del dataloader
torch.cuda.empty_cache()
torch.cuda.synchronize()