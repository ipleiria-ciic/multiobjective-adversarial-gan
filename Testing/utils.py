import os
import scipy
import lpips
import torch
import warnings
import torchvision
import numpy as np
import alive_progress
import matplotlib.pyplot as plt

import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, num_classes=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(torch.nn.utils.spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(torch.nn.utils.spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = torch.nn.utils.spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src
    
# ** Edited by @joseareia on 2025/01/12 **
class AdversarialDiscriminator(nn.Module):
    """Classifier Discriminator to detect adversarial misclassifications."""
    def __init__(self, image_size=128, conv_dim=64, num_classes=5, repeat_num=6):
        """
        A classifier-based discriminator for adversarial training.
        
        Args:
            image_size (int): Size of the input images.
            conv_dim (int): Number of base channels for convolution layers.
            num_classes (int): Number of classification labels.
            repeat_num (int): Number of downsampling layers.
        """
        super(AdversarialDiscriminator, self).__init__()
        layers = []
        layers.append(torch.nn.utils.spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(torch.nn.utils.spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim *= 2
        
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.fc = torch.nn.utils.spectral_norm(nn.Linear(curr_dim * kernel_size * kernel_size, num_classes))

    def forward(self, x):
        """Forward pass for the AdversarialDiscriminator."""
        # Pass the input through the convolutional layers.
        h = self.main(x)
        
        # Flatten the output and apply the final classification layer.
        h = h.view(h.size(0), -1)
        out_class = self.fc(h)
        return out_class
    
class Classifier(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=40, repeat_num=6):
        super(Classifier, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_cls = self.conv2(h)
        return out_cls.view(out_cls.size(0), out_cls.size(1))
    
class Encoder(nn.Module):
    """Encoder network to infer domain vector c from image."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Encoder, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.ReLU())

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.out = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        c_pred = self.out(h)
        return c_pred.view(c_pred.size(0), -1)

class InceptionV3FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        self.inception = torchvision.models.inception_v3(weights='DEFAULT', transform_input=False)
        self.inception.fc = torch.nn.Identity()

    def forward(self, x):
        return self.inception(x)
    
def use_device():
    """
    Gets the device available to use while using Torch.

    Return:
    - device: Device (CPU or GPU).
    """
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return device

def fetch_checkpoint(path, device):
    """
    Fetchs a given checkpoint.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    return checkpoint

def transform_images():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def custom_dataloader(dataroot, batch_size=1):
    """
    A custom Dataloader to load the images generated.
    """
    class CustomDataset(torchvision.datasets.ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            image = self.loader(path)
            if self.transform:
                image = self.transform(image)
            filename = os.path.basename(path)
            class_name = self.classes[target]
            return image, filename, class_name
        
    transform = transform_images()

    dataloader = torch.utils.data.DataLoader(CustomDataset(root=dataroot, transform=transform), batch_size=batch_size, shuffle=True)

    return dataloader

def generate_images(dataloader, attack, delta, netG, netE, device):
    output_dir = f"Testing/Images/Generated-Images-{attack}-Delta-{delta}"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        with alive_progress.alive_bar(len(dataloader), title=f"[ INFO ] Generating adversarial images", bar='classic', spinner=None) as bar:
            for i, (real_img, filename, class_name) in enumerate(dataloader):
                real_img = real_img.to(device)

                c_pred = netE(real_img)

                fake_img = netG(real_img, c_pred).detach().cpu()

                fake_pil = torchvision.transforms.ToPILImage()(fake_img.squeeze(0))

                class_dir = os.path.join(output_dir, class_name[0])
                os.makedirs(class_dir, exist_ok=True)

                save_path = os.path.join(class_dir, filename[0])
                fake_pil.save(save_path)

                bar()

def load_model(name, device):
    model = torch.jit.load(f'Models/{name}.pt')
    model.eval()
    model.to(device)
    return model

def load_dataset(path):
    transform = transform_images()
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    return dataloader

def classify_images(model, dataloader, device, title):
    predictions = {}
    with torch.no_grad():
        with alive_progress.alive_bar(len(dataloader), title=f"[ INFO ] {title}", bar='classic', spinner=None) as bar:
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                for idx, (pred, label) in enumerate(zip(preds.cpu().numpy(), labels.cpu().numpy())):
                    predictions[i * dataloader.batch_size + idx] = (pred, label)
                bar()
    return predictions

def calculate_lpips(original_loader, adversarial_loader, device, attack, delta, model_name):
    warnings.filterwarnings("ignore", module="torch")
    warnings.filterwarnings("ignore", module="lpips")

    lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
    lpips_values = []
    with alive_progress.alive_bar(len(original_loader), title=f"[ INFO ] LPIPS similarity calculation", bar='classic', spinner=None) as bar:
        for (img1, _), (img2, _) in zip(original_loader, adversarial_loader):
            img1, img2 = img1.to(device), img2.to(device)
            lpips_value = lpips_fn(img1, img2).mean().item()
            lpips_values.append(lpips_value)
            bar()
    
    flierprops = dict(marker='o', markerfacecolor='lightgreen', markersize=8, linestyle='none')
    plt.figure(figsize=(6, 6))
    plt.boxplot(lpips_values, vert=True, patch_artist=True, boxprops=dict(facecolor="skyblue"), flierprops=flierprops)
    plt.ylabel("LPIPS Score")
    plt.title(f"LPIPS - {attack} W/Delta {delta} ({model_name})")
    plt.savefig(f"Testing/Plots/LPIPS-Boxplot-{attack}-Delta-{delta}-{model_name}.png")

    return np.mean(lpips_values)

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """
    Calculates the FID score between two Gaussian distributions.

    Parameters:
    - mu1: First mean (usually the original one).
    - sigma1: First covariance (usually the original one).
    - mu2: Second mean (usually the adversarial one).
    - sigma2: Second covariance (usually the adversarial one).

    Return:
    - fid: FID score.
    """

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def get_activations(data_loader, model, device, title):
    """
    Get the activations for an entire Dataloader.

    Parameters:
    - data_loader: A given Dataloader.
    - model: Model to use.
    - device: Device to use (CPU | GPU).

    Return:
    - torch_activations: Activations in a Torch Tensor format.
    """

    model = model.to(device)
    all_activations = []
    with torch.no_grad():
        with alive_progress.alive_bar(len(data_loader), title=f"[ INFO ] {title}", bar='classic', spinner=None) as bar:
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch 
            
                images = images.to(device)
                activations = model(images)
                all_activations.append(activations.cpu())
                bar()
    torch_activations = torch.cat(all_activations, dim=0).numpy()
    torch.cuda.empty_cache()
    return torch_activations

def fid(real_dataset_path, generated_dataset_path, device):
    """
    Preparation of both real and generated datasets for the FID calculation.

    Parameters:
    - real_dataset_path: Real dataset path.
    - generated_dataset_path: Generated (adversarial) dataset path.
    """

    model = InceptionV3FeatureExtractor().eval()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((299, 299)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    real_dataset = torchvision.datasets.ImageFolder(real_dataset_path, transform=transform)
    generated_dataset = torchvision.datasets.ImageFolder(generated_dataset_path, transform=transform)
    
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=8, shuffle=False, num_workers=4)
    generated_loader = torch.utils.data.DataLoader(generated_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Calculate activations for real and generated images.
    real_activations = get_activations(real_loader, model, device, title="Calculating the activations of the real images")
    generated_activations = get_activations(generated_loader, model, device, title="Calculating the activations of the generated images")

    # Calculate mean and covariance.
    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    mu_gen = np.mean(generated_activations, axis=0)
    sigma_gen = np.cov(generated_activations, rowvar=False)

    # Compute FID.
    fid_score = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)

    return fid_score