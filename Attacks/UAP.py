import os
import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
from alive_progress import alive_bar
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description="UAP Attack on Imagewoof Dataset")
parser.add_argument("--model", type=str)
parser.add_argument("--delta", type=str)
args = parser.parse_args()

output_dir = f"Attacks/UAP/UAP-{args.model}-{args.delta}"
os.makedirs(output_dir, exist_ok=True)

class UniversalAdversarialPerturbation:
    def __init__(self, model, epsilon, num_iterations, alpha=1.0):
        self.model = model
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.alpha = alpha
    
    def generate_uap(self, data_loader, device):
        uap = torch.zeros_like(next(iter(data_loader))[0]).to(device)

        uap.requires_grad = True
        
        optimizer = optim.Adam([uap], lr=self.alpha)
        
        for i in range(self.num_iterations):
            total_loss = 0
            for images, _ in data_loader:
                images = images.to(device)
                
                perturbed_images = images + uap
                perturbed_images = torch.clamp(perturbed_images, 0, 1)
                
                outputs = self.model(perturbed_images)
                loss = F.cross_entropy(outputs, torch.argmax(outputs, dim=1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
            print(f"Iteration {i+1}/{self.num_iterations}, Loss: {total_loss:.04f}")
        return uap.detach()

imagewoof_path = "Dataset/Imagewoof/train"
model_path = f"Models/{args.model}.pt"

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=imagewoof_path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

model = torch.jit.load(model_path, map_location=device)
model.to(device).eval()

uap_generator = UniversalAdversarialPerturbation(model, epsilon=args.delta, num_iterations=10)
uap = uap_generator.generate_uap(dataloader, device)

uap_path = f"Attacks/Perturbations/UAP.pt"
os.makedirs(os.path.dirname(uap_path), exist_ok=True)
torch.save(uap, uap_path)

unnormalize = transforms.Compose([
    transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    ),
    transforms.ToPILImage(),
])

del dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

with alive_bar(len(dataloader), title=f"[ INFO ] Generating the UAP attack ({float(args.delta):.02f})", bar='classic', spinner=None) as bar:
    for i, (image_tensor, _) in enumerate(dataloader):
        image_tensor = image_tensor.to(device)
        uap = uap.mean(dim=0, keepdim=True)

        perturbed_image_tensor = image_tensor + uap * float(args.delta)
        perturbed_image_tensor = perturbed_image_tensor.squeeze(0)
        perturbed_image = unnormalize(perturbed_image_tensor)

        output = model(image_tensor) 
        label = output.argmax(dim=1)

        class_folder = dataset.classes[label.item()]
        class_output_folder = os.path.join(output_dir, class_folder)
        os.makedirs(class_output_folder, exist_ok=True)

        original_path, _ = dataset.samples[i]
        filename = os.path.basename(original_path)
        perturbed_image.save(os.path.join(class_output_folder, filename))

        bar()

del model, dataloader
torch.cuda.empty_cache()
torch.cuda.synchronize()