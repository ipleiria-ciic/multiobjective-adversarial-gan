import os
import utils
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import Generator
from model import Encoder

parser = argparse.ArgumentParser()
parser.add_argument("--attack", type=str)
parser.add_argument("--checkpoint_epochs", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--delta", type=float)
args = parser.parse_args()

dataroot = "Dataset/Imagewoof"
adam_beta = 0.5
adam_lr = 0.0002
feature_map_e = 32
feature_map_g = 32
best_epoch = args.checkpoint_epochs
attack = args.attack
num_epochs = 250
batch_size = 128
delta = f"{args.delta:.2f}"

dir = f"SuperstarGAN/models/{attack}/{delta}/Encoder"
os.makedirs(dir, exist_ok=True)

device = utils.use_device()

dataloader = utils.get_dataloader(dataroot, image_size=128, batch_size=batch_size, workers=4)

netE = Encoder(image_size=128, conv_dim=feature_map_e, c_dim=10).to(device)

netG = Generator(conv_dim=feature_map_g, c_dim=10, repeat_num=6).to(device)

checkpoint_name = f"SuperstarGAN/models/{attack}/{delta}/Checkpoint-Epoch-{best_epoch}.pth"
checkpoint = torch.load(checkpoint_name, weights_only=True)
netG.load_state_dict(checkpoint["netG_state_dict"])
netG.to(device)
netG.eval()

criterion = nn.MSELoss()

optimizerE = optim.Adam(netE.parameters(), lr=adam_lr, betas=(adam_beta, 0.999))

encoder_loss = []
best_loss = float('inf')

print(f"[ INFO ] Encoder training for {attack} ({delta})")

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        optimizerE.zero_grad()

        c_pred = netE(real_images)
        reconstructed_images = netG(real_images, c_pred)

        loss = criterion(reconstructed_images, real_images)

        loss.backward()
        optimizerE.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{num_epochs}][Batch {i:03}/{len(dataloader)}] - Loss: {loss.item():.4f}")
        
        encoder_loss.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'epoch': epoch,
                'netE_state_dict': netE.state_dict(),
                'optimizerE_state_dict': optimizerE.state_dict(),
                'best_loss': best_loss,
                'encoder_losses': encoder_loss
            }, f"SuperstarGAN/models/{attack}/{delta}/Encoder/Checkpoint-Epoch-{epoch}.pth")

torch.save({
    'netE_state_dict': netE.state_dict(),
    'optimizerE_state_dict': optimizerE.state_dict(),
    'encoder_losses': encoder_loss
}, f"SuperstarGAN/models/{attack}/{delta}/Encoder/Checkpoint-Epoch-{num_epochs}.pth")

del netG, netE
torch.cuda.empty_cache() 
torch.cuda.synchronize()