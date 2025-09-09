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
from ..loss import TexTokTrainingLoss
from ..ema import EMA
from ..models.model import StyleGANDiscriminator





def set_requires_grad(m, flag: bool):
    for p in m.parameters():
        p.requires_grad = flag






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


		# Hard-coded optimizer (Adam with paper's settings)
		self.optim = torch.optim.Adam(
			self.model.parameters(),
			lr=1e-4,
			betas=(0.0, 0.99),
			weight_decay=0.0
		)

		# Hard-coded scheduler: linear warmup then constant
		self.scheduler = torch.optim.lr_scheduler.LambdaLR(
			self.optim,
			lambda step: min((step + 1) / (0.01 * decay_steps), 1.0)
		)

		# Hard-coded loss function
		self.loss_fn = TexTokTrainingLoss(
			recon_weight=1.0,
			perceptual_weight=0.1,
			gan_weight=0.1,
			lecam_weight=0.0001
		)


		# Initialize discriminator
		self.discriminator = StyleGANDiscriminator(img_size=256).to(self.device)
		
		# Discriminator optimizer
		self.optim_d = torch.optim.Adam(
			self.discriminator.parameters(),
			lr=1e-4,
			betas=(0.0, 0.99),
			weight_decay=0.0
		)
		
		
		# prepare model, optimizer, and dataloader for distributed training
		self.model, self.optim, self.discriminator, self.optim_d, self.scheduler, self.train_dl, self.val_dl = self.accelerator.prepare(
			self.model, 
			self.optim, 
			self.discriminator,
			self.optim_d,
			self.scheduler, 
			self.train_dl, 
			self.val_dl
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

		# load models
		self.resume_from_checkpoint()


	def train(self):
		start_epoch = self.global_step // len(self.train_dl)

		# Training settings
		discriminator_start_step = 80000
		r1_every = 16

			
		for epoch in range(start_epoch, self.num_epoch):
			with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
				for batch in train_dl:
					images, captions = batch
					images = images.to(self.device)
				
					# =========================
					# GENERATOR TRAINING STEP
					# =========================
					with self.accelerator.accumulate(self.model):
						set_requires_grad(self.discriminator, False)  # Freeze discriminator
						
						self.optim.zero_grad(set_to_none=True)
						
						with self.accelerator.autocast():
							recon, tokens = self.model(images, captions)
							
							# Get discriminator predictions for fake images
							if self.global_step >= discriminator_start_step:
								fake_pred = self.discriminator(recon)
								g_loss, g_loss_dict = self.loss_fn.compute_generator_loss(
									images, recon, fake_pred
								)
							else:
								# Only reconstruction loss before discriminator starts
								g_loss = F.mse_loss(recon, images)
								g_loss_dict = {'recon_loss': g_loss.item()}
						
						self.accelerator.backward(g_loss)
						
						if self.accelerator.sync_gradients and self.max_grad_norm:
							self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
						
						self.optim.step()
						
						# # EMA update
						# if self.use_ema and self.accelerator.sync_gradients:
						# 	unwrapped_model = self.accelerator.unwrap_model(self.model)
						# 	self.ema.update(unwrapped_model)

						if self.use_ema and self.accelerator.sync_gradients:
							if self.accelerator.is_main_process:  # Only update EMA on main process
								unwrapped_model = self.accelerator.unwrap_model(self.model)
								self.ema.update(unwrapped_model)
												
						self.scheduler.step()

					# =========================
					# DISCRIMINATOR TRAINING STEP
					# =========================
					d_loss = torch.tensor(0.0, device=self.device)
					d_loss_dict = {}
					
					if self.global_step >= discriminator_start_step:
						with self.accelerator.accumulate(self.discriminator):
							set_requires_grad(self.discriminator, True)  # Unfreeze discriminator
							
							self.optim_d.zero_grad(set_to_none=True)
							
							with self.accelerator.autocast():
								real_pred = self.discriminator(images)
								fake_pred = self.discriminator(recon.detach())  # Detach to avoid generator gradients
								
								# Apply R1 regularization periodically
								apply_r1 = (self.global_step % r1_every == 0)
								
								if apply_r1:
									images_for_r1 = images.clone().detach().requires_grad_(True)
									real_pred_for_r1 = self.discriminator(images_for_r1)
									d_loss, d_loss_dict = self.loss_fn.compute_discriminator_loss(
										real_pred_for_r1, fake_pred, images_for_r1, apply_r1=True
									)
								else:
									d_loss, d_loss_dict = self.loss_fn.compute_discriminator_loss(
										real_pred, fake_pred, apply_r1=False
									)
							
							self.accelerator.backward(d_loss)
							
							if self.accelerator.sync_gradients and self.max_grad_norm:
								self.accelerator.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
							
							self.optim_d.step()

					# =========================
					# LOGGING AND CHECKPOINTING
					# =========================
					if self.accelerator.sync_gradients:
						if not (self.global_step % self.save_every) and self.accelerator.is_main_process:
							self.save_ckpt(rewrite=True)
						
						if not (self.global_step % self.sample_every):
							self.sample_prompts()
						
						# Prepare logging
						log_dict = {
							"g_loss": g_loss.item(),
							"lr": self.optim.param_groups[0]['lr']
						}
						
						# Add generator loss components
						for key, value in g_loss_dict.items():
							log_dict[f"g_{key}"] = value
						
						# Add discriminator losses if training
						if self.global_step >= discriminator_start_step:
							log_dict["d_loss"] = d_loss.item()
							for key, value in d_loss_dict.items():
								log_dict[f"d_{key}"] = value
						
						self.accelerator.log(log_dict, step=self.global_step)
						self.global_step += 1

		self.accelerator.end_training()        
		print("Train finished!")

  
	@torch.no_grad()
	def sample_prompts(self):
		self.model.eval()
		if hasattr(self, 'discriminator'):
			self.discriminator.eval()
		
		with tqdm(self.val_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as val_dl:
			all_originals = []
			all_reconstructions = []
			
			for i, batch in enumerate(val_dl):
				if i >= 4:  # Limit to 4 batches
					break
					
				images, captions = batch
				images = images.to(self.device)
				
				# Use EMA model if available, otherwise use main model
				if self.use_ema and self.ema is not None:
					recon, _ = self.ema.ema_model(images, captions)
				else:
					recon, _ = self.model(images, captions)
				
				# Take first 6 images from each batch
				all_originals.append(images[:6])
				all_reconstructions.append(recon[:6])
			
			if all_originals:
				# Concatenate all images
				originals = torch.cat(all_originals, dim=0)
				reconstructions = torch.cat(all_reconstructions, dim=0)
				
				# Create side-by-side comparison: [orig1, recon1, orig2, recon2, ...]
				comparison = torch.stack([originals, reconstructions], dim=1)
				comparison = comparison.view(-1, *originals.shape[1:])
				
				# Create grid with originals and reconstructions alternating
				grid = make_grid(comparison, nrow=12, normalize=True, value_range=(-1, 1), padding=2)
				
				# Log to wandb
				self.accelerator.log({
					"reconstructions": [wandb.Image(grid, caption="Original (left) vs Reconstructed (right)")]
				}, step=self.global_step)
				
				# Save to disk
				save_image(grid, os.path.join(self.image_saved_dir, f'reconstruction_step_{self.global_step}.png'))
		
		self.model.train()
		if hasattr(self, 'discriminator'):
			self.discriminator.train()




	  


