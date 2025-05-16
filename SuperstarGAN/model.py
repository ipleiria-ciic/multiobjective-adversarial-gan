import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils

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