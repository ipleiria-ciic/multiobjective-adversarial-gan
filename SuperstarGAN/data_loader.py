from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

def get_loader(dataset, dataset_train, dataset_test, transform_mean, transform_std, crop_size=178, image_size=128, batch_size=128, mode='train', num_workers=2):
    transform = []
    mode == 'train' and transform.append(T.RandomHorizontalFlip(p=0.5))
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=transform_mean, std=transform_std))
    transform = T.Compose(transform)

    if mode == 'train':
        dataset = ImageFolder(dataset_train, transform)
    elif mode == 'test':
        dataset = ImageFolder(dataset_test, transform)

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(mode=='train'))
    return data_loader
    
def get_loader_class(dataset, dataset_train, transform_mean, transform_std, crop_size=178, image_size=128, batch_size=128, mode='train', num_workers=2):
    transform = []
    mode == 'train' and transform.append(T.RandomHorizontalFlip(p=0.5))
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.RandomRotation(degrees = (-20,20)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=transform_mean, std=transform_std))
    transform = T.Compose(transform)
    
    dataset = ImageFolder(dataset_train, transform)

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(mode=='train'))
    return data_loader