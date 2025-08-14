import torch
import torch.nn.functional as F
import lpips

# ----------------------------
# Loss wrapper (L2 + LPIPS(opt) + GAN + r1 regularizer)
# ----------------------------

class TexTokLoss:
    def __init__(self, lambda_perc=0.1, lambda_adv=0.1, r1_gamma=10.0, use_lpips=True):
        self.lambda_perc = lambda_perc
        self.lambda_adv = lambda_adv
        self.r1_gamma = r1_gamma
        self.lpips = None
       
       
        self.lpips = lpips.LPIPS(net='vgg').eval()
        for p in self.lpips.parameters():
            p.requires_grad = False
    


    def perceptual(self, recon, real):
        if self.lpips is None:
            return torch.tensor(0.0, device=recon.device)

        # make sure the LPIPS network lives on the same device as the tensors
        self.lpips = self.lpips.to(recon.device)

        # LPIPS wants float32; do NOT use torch.no_grad() here (you need grads!)
        # Also disable autocast for stability.
        return self.lpips(recon.float(), real.float()).mean()


    def gen_loss(self, disc, recon):
        if disc is None: return torch.tensor(0.0, device=recon.device)
        logits_fake = disc(recon)
        return -logits_fake.mean()

    def disc_loss(self, disc, real, recon):
        if disc is None: 
            return torch.tensor(0.0, device=real.device), torch.tensor(0.0, device=real.device)
        logits_real = disc(real.detach())
        logits_fake = disc(recon.detach())
        d_loss = F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean()
        # r1 gradient penalty on real
        real.requires_grad_(True)
        logits_real_r1 = disc(real)
        grad = torch.autograd.grad(
            outputs=logits_real_r1.sum(), inputs=real, create_graph=True
        )[0]
        r1 = (grad.view(grad.size(0), -1).pow(2).sum(dim=1)).mean() * (self.r1_gamma * 0.5)
        real.requires_grad_(False)
        return d_loss + r1, r1.detach()

    def __call__(self, recon, real, disc=None):
        l2 = F.mse_loss(recon, real)
        lp = self.perceptual(recon, real)
        adv = self.gen_loss(disc, recon)
        loss = l2 + self.lambda_perc * lp + self.lambda_adv * adv
        return loss, {"l2": l2.item(), "lpips": float(lp), "g_adv": float(adv)}



