import os
import argparse
import alive_progress

import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description="FGSM Attack on Imagewoof Dataset")
parser.add_argument("--model", type=str)
parser.add_argument("--delta", type=str)
args = parser.parse_args()

delta = float(args.delta)
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imagewoof_path = "Dataset/Imagewoof/train"
model_path = f"Models/{args.model}.pt"
output_dir = f"Attacks/FGSM/FGSM-{args.model}-{args.delta}"
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=imagewoof_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = torch.jit.load(model_path, map_location=device)
model.to(device).eval()

criterion = nn.CrossEntropyLoss()

unnormalize = transforms.Compose([
    transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    ),
    transforms.ToPILImage(),
])

with alive_progress.alive_bar(len(dataloader), title=f"[ INFO ] Generating the FGSM attack (Delta={delta:.02f})", bar='classic', spinner=None) as bar:
    for i, (image_tensor, _) in enumerate(dataloader):
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad = True

        output = model(image_tensor)
        label = output.argmax(dim=1)

        loss = criterion(output, label)
        model.zero_grad()
        loss.backward()

        perturbation = float(delta) * image_tensor.grad.sign()
        perturbed_image_tensor = image_tensor + perturbation
        perturbed_image = unnormalize(perturbed_image_tensor.squeeze(0))

        class_folder = dataset.classes[label.item()]
        class_output_folder = os.path.join(output_dir, class_folder)
        os.makedirs(class_output_folder, exist_ok=True)

        original_path, _ = dataset.samples[i]
        filename = os.path.basename(original_path)

        perturbed_image.save(os.path.join(class_output_folder, filename))

        bar()

del model
torch.cuda.empty_cache()
torch.cuda.synchronize()