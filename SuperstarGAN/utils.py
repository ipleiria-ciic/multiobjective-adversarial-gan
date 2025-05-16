import os
import lpips
import warnings
import numpy as np
import alive_progress

from PIL import Image
from scipy.linalg import sqrtm
from statistics import mean, harmonic_mean, median

import torch
import torch.nn as nn

import torchvision.datasets as TD
from torchvision import models, transforms
from torchvision.transforms import v2 as T
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('jpg', 'png', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        self.inception = models.inception_v3(weights='DEFAULT', transform_input=False)
        self.inception.fc = nn.Identity()

    def forward(self, x):
        return self.inception(x)

def fetch_images(img_original_path, img_adversarial_path, img_to_class, logging=False):
    # Mapping the image names to a class.
    image_to_class_mapping = {}
    with open(img_to_class, 'r') as f:
        for line in f:
            if " -- " not in line:
                continue
            img_name, img_class = line.strip().split(' -- ')
            image_to_class_mapping[img_name] = img_class

    # Get the adversarial images. 
    adversarial_images = set(os.listdir(img_adversarial_path))

    # Filter only the images that also exist in the original folder.
    common_images = [img for img in adversarial_images if os.path.exists(os.path.join(img_original_path, img))]

    if logging:
        print(f"[ INFO ] Number of images fetched: {len(common_images)}.")

    original_images = []
    adversarial_images = []
    mapping_classes = []

    # Populate the arrays with the correct information for each.
    for img_name in common_images:
        original_images.append(Image.open(os.path.join(img_original_path, img_name)))
        adversarial_images.append(Image.open(os.path.join(img_adversarial_path, img_name)))
        img_class = image_to_class_mapping.get(img_name)
        mapping_classes.append(img_class)

    return original_images, adversarial_images, mapping_classes

def custom_transforms():
    data_transforms = {
        'train': T.Compose([
            T.ToImage(), # Convert to tensor, because the image comes has PIL
            T.Resize(size=(128, 128)),
            # T.RandomResizedCrop(size=(128, 128), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomGrayscale(p=0.1),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': T.Compose([
            T.ToImage(), # Convert to tensor, because the image comes has PIL
            T.Resize(size=(128, 128)),
            T.CenterCrop(128),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def process_image(img, transpose=False):
    if transpose:
        img = img.numpy().transpose((1, 2, 0)) # Transform (X, Y, Z) shape
    img = (img - img.min()) / (img.max() - img.min()) # Clip the image to [0, 255] values
    return img

def use_device():
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return device

def img_transform(input_folder, output_folder, threshold):
    """
    Removes a threshold in all the images in a given folder.

    Parameters:
    - input_folder: Input folder that contains all the images to be processed.
    - output_folder: Output folder were the transformed images will be stored. 
    - threshold: Numeric value that represents the threshold that will be cropped from the images.
    """  
    if not (0 <= threshold <= 255):
        raise ValueError("Threshold must be a numeric value between 0 and 255.")

    # Check if the output folder exists. If not, creates one.
    os.makedirs(output_folder, exist_ok=True)

    def remove_black_border(image_path, output_path):
        with Image.open(image_path) as image:
            width, height = image.size
            new_img = image.crop((threshold, threshold, width - threshold, height - threshold))
            new_img.save(output_path)
    
    total_files = sum(1 for filename in os.listdir(input_folder) if filename.endswith(".jpg"))

    with alive_progress.alive_bar(total_files, title="[ INFO ] Image processing", bar='classic', spinner=None) as bar:
        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg"):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                remove_black_border(input_path, output_path)
            bar()

def adversarial_classifier(img_original_path, img_adversarial_path, img_to_class, log_path, model_path):
    """
    Classifies the images into original or adversarial.

    Parameters:
    - img_original_path: Original images path.
    - img_adversarial_path: Adversarial images path.
    - img_to_class: Text file that contains a image name to class mapping.
    - log_path: Log file location.
    - model: Location of the model to use in the classification process.
    """

    # Function to classify a given image.
    def classifier(model, device, img_given):    
        data_transforms = custom_transforms()
        
        img = data_transforms['val'](img_given)
        img = img.unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
        
        torch.cuda.empty_cache() 
        return preds[0]+1

    # Get the device available to use.
    device = use_device()

    # Get the classification model.
    model = torch.jit.load(model_path)
    model.eval()

    # Fetch all the images (original and adversarial) and the mapping classes.
    original_images, adversarial_images, mapping_classes = fetch_images(img_original_path, img_adversarial_path, img_to_class, logging=False)

    # Counters.
    total_images = len(original_images)
    acc_original = 0
    acc_adversarial = 0

    with alive_progress.alive_bar(total_images, title="[ INFO ] Image classification", bar='classic', spinner=None) as bar:
        # Classify both original and adversarial image if it is correct.
        for i, img in enumerate(original_images):
            bar()
            # Reject the adversarial images (only noise images).
            if mapping_classes[i] == '9999':
                continue
            
            # Classification of the original image.
            r = classifier(model, device, img)
            
            # If the classification of the original image is correct, the adversarial image will be tested.
            if r == int(mapping_classes[i]):
                # Debugger. Can be deleted later.
                # print(f"[ INFO ] Classification of the image '{i:04}'. [ORI: {int(mapping_classes[i])}; ADV: {r+1}] [ OK ]")
            
                # Accuracy of original images counter.
                acc_original += 1
                
                # Classification of the adversarial image.
                r_adv = classifier(model, device, adversarial_images[i])
                if r_adv != r:
                    acc_adversarial += 1
           # else:
                # Debugger. Can be deleted later.
                # print(f"[ INFO ] Classification of the image '{i:04}'. [ORI: {int(mapping_classes[i])}; ADV: {r+1}] [ NOK ]")
                # continue

    # Create log file with the following information: datetime, total_images, acc_original and acc_adversarial.
    # os.makedirs(log_path, exist_ok=True)
    # txt_class_path = os.path.join(log_path, "log.txt")
    # with open(txt_class_path, mode="a") as file:
    #     file.write(f"\nClassification created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")
    #     file.write(f"Total images classified: {total_images}.\n")
    #     file.write(f"Original accuracy classification: {((acc_original*100)/total_images):.2f}%\n")
    #     file.write(f"Adversarial accuracy classification (in a total of {acc_original} images): {((acc_adversarial*100)/acc_original):.2f}%\n")
    #     file.write(f"Total adversarial images that fooled the classifier: {acc_adversarial}.\n")
    #     file.write(f"---")

    # print(f"[ INFO ] Classification were written in '{txt_class_path}'.")
    torch.cuda.empty_cache()

    len_original = acc_original
    len_adversarial = acc_adversarial
    acc_original = (acc_original*100)/total_images
    acc_adversarial = (acc_adversarial*100)/len_original

    return total_images, len_original, len_adversarial, acc_original, acc_adversarial

# Function to compute activations for an entire DataLoader.
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
            for images in data_loader:
                images = images.to(device)
                activations = model(images)
                all_activations.append(activations.cpu())
                bar()
    torch_activations = torch.cat(all_activations, dim=0).numpy()
    torch.cuda.empty_cache()
    return torch_activations


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
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def fid(real_dataset_path, generated_dataset_path):
    """
    Preparation of both real and generated datasets for the FID calculation.

    Parameters:
    - real_dataset_path: Real dataset path.
    - generated_dataset_path: Generated (adversarial) dataset path.
    """

    model = InceptionV3FeatureExtractor().eval()
    device = use_device()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    real_dataset = ImageDataset(image_dir=real_dataset_path, transform=transform)
    generated_dataset = ImageDataset(image_dir=generated_dataset_path, transform=transform)

    real_loader = DataLoader(real_dataset, batch_size=8, shuffle=False, num_workers=4)
    generated_loader = DataLoader(generated_dataset, batch_size=8, shuffle=False, num_workers=4)

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

def calculate_lpips(real_dataset_path, generated_dataset_path, network='vgg', aggregation='mean'):
    """
    Calculate LPIPS similarity for the entire dataset.

    Parameters:
    - real_dataset_path: Real dataset path.
    - generated_dataset_path: Generated (adversarial) dataset path.
    - network: Network to use in the LPIPS calculation. Default: VGG.
    - aggregation: Aggregation method for final similarity. Options: 'mean', 'harmonic_mean', 'median'. Default: 'mean'.
    """

    # Disable the warning from torch (built-in in LPIPS).
    warnings.filterwarnings("ignore", module="torch")
    warnings.filterwarnings("ignore", module="lpips")

    loss_fn = lpips.LPIPS(net=network, verbose=False)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    real_images = sorted(os.listdir(real_dataset_path))
    generated_images = sorted(os.listdir(generated_dataset_path))

    # Align dataset sizes: the size must the same.
    if len(real_images) > len(generated_images):
        real_images = real_images[:len(generated_images)]
    else:
        generated_images = generated_images[:len(real_images)]

    # Ensure both datasets have the same number of images.
    assert len(real_images) == len(generated_images), "Datasets must have the same number of images."

    # Calculate LPIPS for each image pair.
    lpips_scores = []
    with alive_progress.alive_bar(len(real_images), title=f"[ INFO ] LPIPS similarity calculation", bar='classic', spinner=None) as bar:
        for real_img_name, gen_img_name in zip(real_images, generated_images):
            real_img_path = os.path.join(real_dataset_path, real_img_name)
            gen_img_path = os.path.join(generated_dataset_path, gen_img_name)

            real_img = transform(Image.open(real_img_path).convert("RGB")).unsqueeze(0)
            gen_img = transform(Image.open(gen_img_path).convert("RGB")).unsqueeze(0)

            # Calculate LPIPS similarity.
            similarity = loss_fn(real_img, gen_img).item()
            lpips_scores.append(similarity)

            bar()

    if aggregation == 'mean':
        final_score = mean(lpips_scores)
    elif aggregation == 'harmonic_mean':
        final_score = harmonic_mean(lpips_scores)
    elif aggregation == 'median':
        final_score = median(lpips_scores)
    else:
        raise ValueError("[ ERROR ] Invalid aggregation method. Choose from 'mean', 'harmonic_mean', or 'median'.")
    return final_score, aggregation

def get_dataloader(dataroot, image_size, batch_size, workers):
    """
    Prepare a dataloader for a given dataroot.

    Parameters:
    - dataroot: Path to the dataset.
    - image_size: Image size.
    - batch_size: Batch size.
    - workers: Number of workers.

    Return:
    - dataloader: Dataloader for a given dataset.
    """
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = TD.ImageFolder(root=dataroot, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    return dataloader