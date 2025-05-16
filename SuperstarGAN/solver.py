import os
import time
import datetime
import numpy as np
import alive_progress
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from model import Generator
from model import Discriminator
from model import AdversarialDiscriminator
from model import Classifier

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, loader, class_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.loader = loader
        self.class_loader = class_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.c_conv_dim = config.c_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.c_repeat_num = config.c_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_perturbation = config.lambda_perturbation

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.c_lr = config.c_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.c_beta1 = config.c_beta1
        self.resume_iters = config.resume_iters
        self.nadir_slack = config.nadir_slack
        self.disc_weights = config.disc_weights
        self.delta = f"{config.delta:.02f}"
        self.attack = config.attack

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.result_dir = config.result_dir
        self.model_save_dir = config.model_save_dir
        self.checkpoint_save_dir = f"{self.model_save_dir}/{self.attack}/{self.delta}"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_save_dir, exist_ok=True)
        
        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    # ** Edited by @joseareia on 2024/12/13 **
    # Changelog: Added the list of discriminators and refactored the optimizers call for each discriminator.
    def build_model(self):
        """Create a generator and discriminators."""

        # Create the generator
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        
        # Add a list of discriminators
        self.discriminators = []
        self.discriminators.append(Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num))
        self.discriminators.append(AdversarialDiscriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num))
        
        # Create the classifier
        self.C = Classifier(self.image_size, self.c_conv_dim, self.c_dim, self.c_repeat_num)

        # Optimizers for each network
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizers = []  # Store optimizers for all discriminators
        for discriminator in self.discriminators:
            optimizer = torch.optim.Adam(discriminator.parameters(), self.d_lr, [self.beta1, self.beta2])
            self.d_optimizers.append(optimizer)
        
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.c_lr, [self.c_beta1, self.beta2])

        # Move all models to the specified device
        self.G.to(self.device)
        for discriminator in self.discriminators:
            discriminator.to(self.device)
        self.C.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('[ INFO ] Loading the trained models from step {}.'.format(resume_iters))
        
        ch_path = f"SuperstarGAN/models/{self.attack}/{self.delta}/Checkpoint-Epoch-{self.test_iters}.pth"
        checkpoint = torch.load(ch_path, map_location=lambda storage, loc: storage, weights_only=False)

        self.G.load_state_dict(checkpoint["netG_state_dict"])
        self.discriminators[0].load_state_dict(checkpoint["netD_state_dict"])
        self.discriminators[1].load_state_dict(checkpoint["netD_adv_state_dict"])
        self.C.load_state_dict(checkpoint["netC_state_dict"])

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    # ** Edited by @joseareia on 2024/12/16 **
    # Changelog: Iterate for the list of optimizers in the discriminator.
    def update_lr(self, g_lr, d_lr, c_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr

        for d_optimizer in self.d_optimizers:
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = d_lr

        for param_group in self.c_optimizer.param_groups:
            param_group['lr'] = c_lr

    # ** Edited by @joseareia on 2024/12/16 **
    # Changelog: Iterate for the list of optimizers in the discriminator.
    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()

        for d_optimizer in self.d_optimizers:
            d_optimizer.zero_grad()

        self.c_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=weight, retain_graph=True, create_graph=True, only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, binary=True):
        if binary:
            return F.cross_entropy(logit, target)
        else:
            return F.binary_cross_entropy_with_logits(logit, target, reduction='sum') / logit.size(0)

    def train(self):
        """Train StarGAN within a single dataset."""
        data_loader = self.loader
        data_loader_class = self.class_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim)
        data_iter_class = iter(data_loader_class)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr
        c_lr = self.c_lr

        # ** Edited by @joseareia on 2024/12/16 **
        # Changelog: Print the network used for train, including all the discriminators.
        # self.print_network(self.G, 'G')

        # for discriminator in self.discriminators:
        #     self.print_network(discriminator, 'D')

        # self.print_network(self.C, 'C') 

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('[ INFO ] Training started!')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            try:
                x_real_class, label_org_class = next(data_iter_class)
            except:
                data_iter_class = iter(data_loader_class)
                x_real_class, label_org_class = next(data_iter_class)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = self.label2onehot(label_org, self.c_dim)
            c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device) # Input images.
            x_real_class = x_real_class.to(self.device)

            c_org = c_org.to(self.device) # Original domain labels.
            c_trg = c_trg.to(self.device) # Target domain labels.
            
            label_org = label_org.to(self.device) # Labels for computing classification loss.
            label_trg = label_trg.to(self.device) # Labels for computing classification loss.
            label_org_class = label_org_class.to(self.device)

            # =================================================================================== #
            #                             2-0. Train the Classifier                               #
            # =================================================================================== # 

            # Compute loss with real images.
            out_cls = self.C(x_real_class)

            c_loss = self.classification_loss(out_cls, label_org_class)

            self.reset_grad()
            c_loss.backward(retain_graph=True)
            self.c_optimizer.step()

            # Logging.
            loss = {}
            loss['C/loss'] = c_loss.item()

            # =================================================================================== #
            #                             2-1. Train the discriminators                           #
            # =================================================================================== #

            # ** Edited by @joseareia **
            # Changelog (2024/12/16): Train a list of various discriminators.
            # Changelog (2025/01/20): Add the adversarial discriminator training logic.
            # Changelog (2025/01/23): Refactor the missclassification loss and add perturbation penalty.
            losses_real, losses_fake, losses_gp = [], [], []
            for d_idx, discriminator in enumerate(self.discriminators):
                # Original discriminator. The logic remains unchanged from the original code.
                if d_idx == 0:
                    # Compute loss with real images.
                    out_src = discriminator(x_real)
                    d_loss_real = torch.mean(F.relu(1.0 - out_src))
                    losses_real.append(d_loss_real)

                    # Compute loss with fake images.
                    x_fake = self.G(x_real, c_trg)
                    out_src = discriminator(x_fake.detach())
                    d_loss_fake = torch.mean(F.relu(1.0 + out_src))
                    losses_fake.append(d_loss_fake)

                    # Compute loss for gradient penalty.
                    alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                    x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                    out_src = discriminator(x_hat)
                    d_loss_gp = self.gradient_penalty(out_src, x_hat)
                    losses_gp.append(d_loss_gp)
                else:
                    # Adversarial discriminator logic.
                    x_fake = self.G(x_real, c_trg)

                    # Compute classification loss for adversarial misclassification.
                    out_class_real = discriminator(x_real_class)
                    out_class_fake = discriminator(x_fake.detach())

                    # Loss to penalise correct classification for fake images.
                    misclassification_loss = -self.classification_loss(out_class_fake, label_trg)
                    
                    # Perturbation loss to minimize the difference between real and fake images.
                    perturbation_loss = torch.mean((x_real - x_fake) ** 2)

                    # Weighted combination of the two losses.
                    d_loss = misclassification_loss + self.lambda_perturbation * perturbation_loss

                    losses_real.append(torch.tensor(0.0).to(self.device))   # No real loss for classifier discriminator.
                    losses_fake.append(misclassification_loss)              # Adversarial misclassification loss.
                    losses_gp.append(torch.tensor(0.0).to(self.device))     # No gradient penalty for classifier discriminator.

                if d_idx == 0:
                    # Standard discriminator: include real, fake, and gradient penalty.
                    d_loss = losses_real[d_idx] + losses_fake[d_idx] + self.lambda_gp * losses_gp[d_idx]
                else:
                    # Adversarial classifier discriminator: already computed as `d_loss`.
                    pass

                # Backward and optimize.
                self.reset_grad()
                d_loss.backward()
                self.d_optimizers[d_idx].step()

            # Weighted general loss for all discriminators (if needed for logging).
            # d_loss_general = (
            #     self.disc_weights[0] * (losses_real[0] + losses_fake[0] + self.lambda_gp * losses_gp[0]) +
            #     self.disc_weights[1] * (losses_fake[1])
            # )

            # Logging the general discriminator loss components.
            loss['D/Loss_Real'] = (float(self.disc_weights[0]) * losses_real[0].item())
            loss['D/Loss_Fake'] = (float(self.disc_weights[0]) * losses_fake[0].item() + float(self.disc_weights[1]) * losses_fake[1].item())


            # =================================================================================== #
            #                               2-2. Train the generator                              #
            # =================================================================================== #

            # ** Edited by @joseareia **
            # Changelog (2024/12/16): Update the train of the generator to include all the losses from the all discriminators.
            # Changelog (2025/01/21): Add perturbation lambda and penalty values to the generator calculation loss.
            # Changelog (2025/01/25): Refactor the weighted adversarial losses by both discriminators.
            if (i+1) % self.n_critic == 0:
                # Generate fake images.
                x_fake = self.G(x_real, c_trg)

                # Compute adversarial losses weighted by discriminator contributions.
                weighted_adversarial_losses = []
                for d_idx, (weight, discriminator) in enumerate(zip(self.disc_weights, self.discriminators)):
                    weight = float(weight)
                    if d_idx == 0:
                         # Standard discriminator: adversarial loss based on raw output.
                        weighted_adversarial_losses.append(weight * torch.mean(discriminator(x_fake)))
                    else:
                        # Adversarial classifier discriminator: focus on target class misclassification.
                        out_class_fake = discriminator(x_fake)
                
                        # This is negative because the generator wants to maximize this loss
                        misclassification_score = -self.classification_loss(out_class_fake, c_trg) 
                        weighted_adversarial_losses.append(weight * misclassification_score)

                # Update nadir point using the weighted adversarial losses.
                self.update_nadir([loss.item() for loss in weighted_adversarial_losses])

                # Hypervolume calculation.
                hypervolume = -torch.sum(torch.stack([torch.log(torch.clamp(self.nadir - loss, min=1e-8)) for loss in weighted_adversarial_losses]))

                # Classification loss.
                out_cls_f = self.C(x_fake)
                c_loss_f = self.classification_loss(out_cls_f, c_trg)

                # Reconstruction loss (target-to-original domain).
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Perturbation penalty (minimising changes to real images).
                perturbation_penalty = torch.mean((x_real - x_fake) ** 2)

                # Combine all generator losses.
                g_loss = (
                    hypervolume 
                    + self.lambda_rec * g_loss_rec 
                    + self.lambda_cls * c_loss_f 
                    + self.nadir 
                    + self.lambda_perturbation * perturbation_penalty
                )

                # Backward and optimize.
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging for generator losses.
                loss['G/Hypervolume'] = hypervolume.item()
                loss['G/Loss_Rec'] = self.lambda_rec * g_loss_rec.item()
                loss['G/Loss_Cls'] = self.lambda_cls * c_loss_f.item()
                loss['G/Nadir'] = self.nadir
                loss['G/Perturbation'] = self.lambda_perturbation * perturbation_penalty.item()

            # =================================================================================== #
            #                                 3. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                elapsed_time = time.time() - start_time
                formatted_time = str(datetime.timedelta(seconds=elapsed_time))[:-7]

                log = f"Epochs [{i + 1}/{self.num_iters}] [{formatted_time}]"
                loss_details = ", ".join(f"{tag}: {value:.4f}" for tag, value in loss.items())
                
                log += f", {loss_details}"
                print(log)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir,f'Sample-{self.attack}-{self.delta}-{i+1}.jpg')
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print("[ INFO ] Saving images into '{}'.".format(sample_path))

            # ** Edited by @joseareia on 2025/04/27 **
            # Changelog: Update the method of saving models to include all the discriminator models.
            if (i+1) % self.model_save_step == 0:
                torch.save({
                    'netG_state_dict': self.G.state_dict(),
                    'netC_state_dict': self.C.state_dict(),
                    'netD_state_dict': self.discriminators[0].state_dict(),
                    'netD_adv_state_dict': self.discriminators[1].state_dict(),
                    'losses': loss,
                }, f'{self.checkpoint_save_dir}/Checkpoint-Epoch-{i+1}.pth')

                print("[ INFO ] Saving checkpoints into '{}'".format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                c_lr -= (self.c_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr, c_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}, c_lr: {}.'.format(g_lr, d_lr, c_lr))

        torch.save({
            'netG_state_dict': self.G.state_dict(),
            'netC_state_dict': self.C.state_dict(),
            'netD_state_dict': self.discriminators[0].state_dict(),
            'netD_adv_state_dict': self.discriminators[1].state_dict(),
            'losses': loss,
        }, f'{self.checkpoint_save_dir}/Checkpoint-Epoch-{i+1}.pth')
        
        del self.G
        del self.C
        for d in self.discriminators:
            del d
        torch.cuda.empty_cache()

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        data_loader = self.loader

        # ** Edited by @joseareia **
        # Changelog (2024/01/30): Add alive progress animation.
        with torch.no_grad():
            with alive_progress.alive_bar(len(data_loader), title=f"[ INFO ] Extracting images", bar='classic', spinner=None) as bar:
                for i, (x_real, c_org) in enumerate(data_loader):
                    # Prepare input images and target domain labels.
                    x_real = x_real.to(self.device)

                    # Create class target lists.
                    c_trg_list = self.create_labels(c_org, self.c_dim)

                    # Translate images.
                    x_fake_list = []
                    
                    for c_trg in c_trg_list:
                        x_fake_list.append(self.G(x_real, c_trg))
                        break # Only the first attribute

                    for i, x_fake in enumerate(x_fake_list[0]):
                        result_path = os.path.join(self.result_dir, f"Image-{i}.jpg")
                        image = x_fake.cpu()
                        save_image(self.denorm(image), result_path)
                    bar()

    # ** Created by @joseareia 2025/01/23 **
    def update_nadir(self, losses_list):
            # Update nadir point dynamically with slack and a small constant for stability.
            self.nadir = float(np.max(losses_list) * self.nadir_slack + 1e-8)

class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):
        hinge_loss = 1. - torch.mul(output, target)
        return torch.mean(F.relu(hinge_loss))