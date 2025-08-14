import torch.nn as nn
import torch.nn.functional as F
import torch


# Loss functions
class TexTokGeneratorLoss(nn.Module):
    """Generator loss for TexTok training"""
    def __init__(self, 
                 recon_weight=1.0,
                 perceptual_weight=0.1,
                 gan_weight=0.1):
        super().__init__()
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight
        
        # Perceptual loss (VGG features)
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features
        self.perceptual_net = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.perceptual_net.parameters():
            param.requires_grad = False
            
    def perceptual_loss(self, input, target):
        # Normalize to [0,1] range for VGG
        input_norm = (input + 1) / 2  # Assuming input is in [-1,1]
        target_norm = (target + 1) / 2
        
        input_features = self.perceptual_net(input_norm)
        target_features = self.perceptual_net(target_norm)
        return F.mse_loss(input_features, target_features)
        
    def forward(self, input_images, reconstructed, discriminator_fake_logits):
        # L2 Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, input_images)
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(reconstructed, input_images)
        
        # Non-saturating GAN loss (generator wants to fool discriminator)
        gan_loss = F.softplus(-discriminator_fake_logits).mean()
        
        # Total generator loss
        total_loss = (self.recon_weight * recon_loss + 
                     self.perceptual_weight * perceptual_loss + 
                     self.gan_weight * gan_loss)
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'gan_loss': gan_loss.item(),
            'total_loss': total_loss.item()
        }


class TexTokDiscriminatorLoss(nn.Module):
    """Discriminator loss for TexTok training with R1 regularization and LeCAM"""
    def __init__(self, r1_weight=10.0, lecam_weight=0.0001):
        super().__init__()
        self.r1_weight = r1_weight
        self.lecam_weight = lecam_weight
        
    def r1_penalty(self, real_pred, real_img):
        """R1 gradient penalty"""
        grad_real = torch.autograd.grad(
            outputs=real_pred.sum(), 
            inputs=real_img, 
            create_graph=True, 
            retain_graph=True
        )[0]
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty
    
    def lecam_regularization(self, real_pred, fake_pred):
        """LeCAM regularization from the paper"""
        real_mean = real_pred.mean()
        fake_mean = fake_pred.mean()
        lecam_loss = torch.pow(real_mean - fake_mean, 2)
        return lecam_loss
        
    def forward(self, real_pred, fake_pred, real_img=None, apply_r1=False):
        # Standard GAN loss (discriminator wants to classify correctly)
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()
        gan_loss = real_loss + fake_loss
        
        total_loss = gan_loss
        loss_dict = {
            'disc_real_loss': real_loss.item(),
            'disc_fake_loss': fake_loss.item(),
            'disc_gan_loss': gan_loss.item()
        }
        
        # R1 gradient penalty (applied periodically during training)
        if apply_r1 and real_img is not None:
            r1_penalty = self.r1_penalty(real_pred, real_img)
            r1_loss = self.r1_weight * r1_penalty
            total_loss = total_loss + r1_loss
            loss_dict['r1_penalty'] = r1_penalty.item()
            loss_dict['r1_loss'] = r1_loss.item()
        
        # LeCAM regularization
        if self.lecam_weight > 0:
            lecam_loss = self.lecam_weight * self.lecam_regularization(real_pred, fake_pred)
            total_loss = total_loss + lecam_loss
            loss_dict['lecam_loss'] = lecam_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class TexTokTrainingLoss(nn.Module):
    """Combined training loss handler for TexTok"""
    def __init__(self, 
                 recon_weight=1.0,
                 perceptual_weight=0.1,
                 gan_weight=0.1,
                 r1_weight=10.0,
                 lecam_weight=0.0001):
        super().__init__()
        
        self.generator_loss = TexTokGeneratorLoss(
            recon_weight=recon_weight,
            perceptual_weight=perceptual_weight,
            gan_weight=gan_weight
        )
        
        self.discriminator_loss = TexTokDiscriminatorLoss(
            r1_weight=r1_weight,
            lecam_weight=lecam_weight
        )
    
    def compute_generator_loss(self, input_images, reconstructed, discriminator_fake_logits):
        return self.generator_loss(input_images, reconstructed, discriminator_fake_logits)
    
    def compute_discriminator_loss(self, real_pred, fake_pred, real_img=None, apply_r1=False):
        return self.discriminator_loss(real_pred, fake_pred, real_img, apply_r1)
