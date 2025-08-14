import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from .utils.scheduler import get_scheduler
from .utils.optimizer import get_optimizer
import wandb
from tqdm import tqdm
# import constant_learnign rate swith warm up
import logging
from .utils.base_trainer import BaseTrainer
from ..loss import TexTokLoss
from ..ema import EMA



def set_requires_grad(m, flag: bool):
    for p in m.parameters():
        p.requires_grad = flag

# ----------------------------
# Optional: a tiny PatchGAN-style discriminator (very light)
# ----------------------------

class TinyPatchGAN(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        # Downsample to a patch map of logits
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.net = nn.Sequential(
            block(in_ch, base),
            block(base, base*2),
            block(base*2, base*4),
            nn.Conv2d(base*4, 1, 3, 1, 1)
        )
    def forward(self, x):
        return self.net(x)  # (B,1,h',w')




class TexTokTrainer(BaseTrainer):
	def __init__(
		self, 
		cfg,
		model,
		dataloaders
		):
		super().__init__(cfg, model, dataloaders)
  
		# Training parameters
		decay_steps = cfg.lr_scheduler.params.decay_steps

  
		if not decay_steps:
			decay_steps = self.num_training_steps


		self.optim = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.0, 0.99))

		self.disc = TinyPatchGAN().to(self.device)


		self.loss_func = TexTokLoss(lambda_perc=0.1, lambda_adv=0.1, r1_gamma=10.0, use_lpips=True)


		self.disc_optim = torch.optim.AdamW(self.disc.parameters(), lr=1e-4, betas=(0.0, 0.99))


	
		

		# self.scheduler = get_scheduler(cfg, self.optim, decay_steps=decay_steps)

		# prepare model, optimizer, and dataloader for distributed training
		self.model, self.disc, self.optim, self.disc_optim, self.train_dl, self.val_dl = \
			self.accelerator.prepare(
				self.model, self.disc, self.optim, self.disc_optim, self.train_dl, self.val_dl
			)
		

		self.use_ema = True
		if self.use_ema:
			# Get the unwrapped model to ensure we copy the correct architecture
			unwrapped_model = self.accelerator.unwrap_model(self.model)
			self.ema = EMA(
				model=unwrapped_model,  # Use unwrapped model
				decay=getattr(cfg, 'ema_decay', 0.999),
				update_after=getattr(cfg, 'ema_update_after', 100)
			)
		else:
			self.ema = None







	def train(self):
		start_epoch=self.global_step//len(self.train_dl)
	  
		for epoch in range(start_epoch, self.num_epoch):
			with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
				for batch in train_dl:
					images, captions = batch

					images = images.to(self.device)
				
					with self.accelerator.accumulate(self.model):
						with self.accelerator.autocast():
							recon = self.model(images, captions)
						
						# G step
						# compute loss
						# set_requires_grad(self.disc, False) 
						self.optim.zero_grad(set_to_none=True)
						with self.accelerator.autocast():
							g_loss, _  = self.loss_func(recon, images, disc=None)
						self.accelerator.backward(g_loss)
						if self.accelerator.sync_gradients and self.max_grad_norm:
							self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
						self.optim.step()
						          
						# 🔥 EMA UPDATE - This is the key addition!
						if self.use_ema and self.accelerator.sync_gradients:
							# Update EMA after optimizer step
							unwrapped_model = self.accelerator.unwrap_model(self.model)
							self.ema.update(unwrapped_model)

						# self.scheduler.step(self.global_step)
					


						# # D step (optional)
						# set_requires_grad(self.disc, True)  
						# self.disc_optim.zero_grad(set_to_none=True)
						# d_loss, r1 = loss_dict['d_loss'](self.disc, images, recon.detach())
						# self.accelerator.backward(d_loss)
						# if self.accelerator.sync_gradients:
						# 	self.disc_optim.step()

					if self.accelerator.sync_gradients:

						if not (self.global_step % self.save_every):
							self.save_ckpt(rewrite=True)
						
						if not (self.global_step % self.sample_every):
							self.sample_prompts()
      
						# if not (self.global_step % self.eval_every):
						# 	self.evaluate()
						lr = self.optim.param_groups[0]['lr']
						self.accelerator.log({"g_loss": g_loss.item(), 
												# "d_loss": d_loss.item(),
												# "r1": r1.item(),
												"lr": lr}, step=self.global_step)
						self.global_step += 1
	  
					
		self.accelerator.end_training()        
		print("Train finished!")
	
	# @torch.no_grad()
	# def sample_prompts(self):
	# 	logging.info("Sampling prompts")
	# 	self.model.eval()
	# 	prompts = []
	# 	with open("data/prompts/dalle_prompts.txt", "r") as f:
	# 		for line in f:
	# 			text = line.strip()
	# 			prompts.append(text)
	# 	imgs = self.model.generate(prompts)
	# 	grid = make_grid(imgs, nrow=6, normalize=False, value_range=(-1, 1))
	# 	# send this to wandb
	# 	self.accelerator.log({"samples": [wandb.Image(grid, caption="Generated samples")]})
	# 	save_image(grid, os.path.join(self.image_saved_dir, f'step.png'))
	# 	self.model.train()
  
  
	@torch.no_grad()
	def sample_prompts(self):
		self.model.eval()
		with tqdm(self.val_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as val_dl:
			for i, batch in enumerate(val_dl):
				image, captions = batch
				image = image.to(self.device)

				if i > 3:
					break
					
				#  use the ema model
				if self.use_ema:
					# Use EMA model for inference
					recon = self.ema.ema_model(image, captions)
				else:
					# Use the original model
					# recon = self.model(image, captions)
					recon = self.model(image, captions)
				# recon = self.model(image, captions)

				grid = make_grid(recon, nrow=6, normalize=True, value_range=(-1, 1))
				# grid = make_grid(recon, nrow=6, normalize=False)
				# send this to wandb
				self.accelerator.log({"samples": [wandb.Image(grid, caption="Generated samples")]})
				save_image(grid, os.path.join(self.image_saved_dir, f'step_{i}.png'))
		self.model.train()






	  


