import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import wandb
from tqdm import tqdm
# import constant_learnign rate swith warm up
import logging
from texttok.loss import TexTokTrainingLoss
from texttok.model import TexTokVAE
from texttok.ema import EMA
from texttok.model import StyleGANDiscriminator
import math

from transformers import get_cosine_schedule_with_warmup
from datasets.coco import get_coco_loaders

import os
import torch
import random
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from einops import rearrange
import logging



def set_requires_grad(m, flag: bool):
    for p in m.parameters():
        p.requires_grad = flag


def resume_from_checkpoint(filename):
		checkpoint = torch.load(filename, map_location=device, weights_only=False)
		global_step = checkpoint['step']
		model.load_state_dict(checkpoint['g_state_dict'])
		discriminator.load_state_dict(checkpoint['d_state_dict'])
		optim.load_state_dict(checkpoint['g_optimizer_state_dict'])
		optim_d.load_state_dict(checkpoint['d_optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		ema.ema_model.load_state_dict(checkpoint['ema_state_dict'])
		
		logging.info(f"Resumed from checkpoint: {checkpoint_path} at step {global_step}")

		return global_step


def save_ckpt(accelerator, model, discriminator, optim, optim_d, scheduler, ema, global_step, filename):
	checkpoint={
			'step': global_step,
			'g_state_dict': accelerator.unwrap_model(model).state_dict(),
			'd_state_dict': accelerator.unwrap_model(discriminator).state_dict(),
			'g_optimizer_state_dict':optim.state_dict(),
			'd_optimizer_state_dict':optim_d.state_dict(),
			'scheduler_state_dict': scheduler.state_dict(),
			'ema_state_dict': ema.ema_model.state_dict(),

		}
	accelerator.save(checkpoint, filename)
	logging.info("Saving checkpoint: %s ...", filename)


@torch.no_grad()
def sample_prompts(model, val_dl, accelerator, device, global_step, use_ema=False, ema=None):
	model.eval()
	
	
	with tqdm(val_dl, dynamic_ncols=True, disable=not accelerator.is_main_process) as val_dl:
		all_originals = []
		all_reconstructions = []
		
		for i, batch in enumerate(val_dl):
			if i >= 4:  # Limit to 4 batches
				break
				
			images, captions = batch
			images = images.to(device)
			
			# Use EMA model if available, otherwise use main model
			if use_ema and ema is not None:
				recon, _ = ema.ema_model(images, captions)
			else:
				recon, _ = model(images, captions)
			
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
			accelerator.log({
				"reconstructions": [wandb.Image(grid, caption="Original (left) vs Reconstructed (right)")]
			}, step=global_step)
			
			# # Save to disk
			# save_image(grid, os.path.join(image_saved_dir, f'reconstruction_step_{global_step}.png'))
	


def train(args):

	global_step = 0

	# setup accelerator
	accelerator = Accelerator(
		mixed_precision=args.mixed_precision,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		log_with="wandb")

	accelerator.init_trackers(
			project_name=args.project_name,
			# add kwargs for wandb
			init_kwargs={"wandb": {
				"config": vars(args)
			}}	
	)

	# set device
	device = accelerator.device
	# model
	model = TexTokVAE()
	# Train loders
	train_dl, val_dl = get_coco_loaders(
		root=args.root,
		batch_size=args.batch_size,
		# shuffle=True,
		num_workers=args.num_workers
	)


	# training parameters
	optim = torch.optim.Adam(
		model.parameters(),
		lr=1e-4,
		betas=(0.0, 0.99),
		weight_decay=0.0
	)
	steps_per_epoch = len(train_dl) // args.gradient_accumulation_steps
	num_training_steps = args.num_epochs * steps_per_epoch
	scheduler = get_cosine_schedule_with_warmup(
			optim,
			num_warmup_steps=4000,
			num_training_steps=num_training_steps
		)

	loss_fn = TexTokTrainingLoss(
		recon_weight=1.0,
		perceptual_weight=0.1,
		gan_weight=0.1,
		lecam_weight=0.0001
	)


	# Initialize discriminator
	discriminator = StyleGANDiscriminator(img_size=256).to(device)
	# Discriminator optimizer
	optim_d = torch.optim.Adam(
		discriminator.parameters(),
		lr=1e-4,
		betas=(0.0, 0.99),
		weight_decay=0.0
	)


	# prepare model, optimizer, and dataloader for distributed training
	model, optim, discriminator, optim_d, scheduler, train_dl, val_dl = accelerator.prepare(
		model, 
		optim, 
		discriminator,
		optim_d,
		scheduler, 
		train_dl, 
		val_dl
	)
	
	if args.use_ema:
		# Get the unwrapped model to ensure we copy the correct architecture
		unwrapped_model = accelerator.unwrap_model(model)
		ema = EMA(
			model=unwrapped_model,  # Use unwrapped model
			decay=args.ema_decay,
			update_after=args.ema_update_after
		)
	else:
		ema = None

	# load models
	if args.resume:
		global_step = resume_from_checkpoint(args.resume)

	effective_steps_per_epoch = math.ceil(len(train_dl) / args.gradient_accumulation_steps)
	effective_training_steps = args.num_epochs * effective_steps_per_epoch

	logging.info(f"Effective batch size per device: {args.batch_size * args.gradient_accumulation_steps}")
	logging.info(f"Effective Total training steps: {effective_training_steps}")

	start_epoch = global_step // len(train_dl)

		
	for epoch in range(start_epoch, args.num_epochs):
		with tqdm(train_dl, dynamic_ncols=True, disable=not accelerator.is_main_process) as train_dl:
			for batch in train_dl:
				images, captions = batch
				images = images.to(device)
			
				# =========================
				# GENERATOR TRAINING STEP
				# =========================
				with accelerator.accumulate(model):
					set_requires_grad(discriminator, False)  # Freeze discriminator
					
					optim.zero_grad(set_to_none=True)
					
					with accelerator.autocast():
						recon, tokens = model(images, captions)
						
						# Get discriminator predictions for fake images
						if global_step >= args.discriminator_start_step:
							fake_pred = discriminator(recon)
							g_loss, g_loss_dict = loss_fn.compute_generator_loss(
								images, recon, fake_pred
							)
						else:
							# Only reconstruction loss before discriminator starts
							g_loss = F.mse_loss(recon, images)
							g_loss_dict = {'recon_loss': g_loss.item()}
					
					accelerator.backward(g_loss)
					
					if accelerator.sync_gradients and args.max_grad_norm:
						accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
					
					optim.step()
					
			
					if args.use_ema and accelerator.sync_gradients:
						if accelerator.is_main_process:  # Only update EMA on main process
							unwrapped_model = accelerator.unwrap_model(model)
							ema.update(unwrapped_model)
											
					scheduler.step()

				# =========================
				# DISCRIMINATOR TRAINING STEP
				# =========================
				d_loss = torch.tensor(0.0, device=device)
				d_loss_dict = {}
				
				if global_step >= args.discriminator_start_step:
					with accelerator.accumulate(discriminator):
						set_requires_grad(discriminator, True)  # Unfreeze discriminator
						
						optim_d.zero_grad(set_to_none=True)
						
						with accelerator.autocast():
							real_pred = discriminator(images)
							fake_pred = discriminator(recon.detach())  # Detach to avoid generator gradients
							
							# Apply R1 regularization periodically
							apply_r1 = (global_step % r1_every == 0)
		
							if apply_r1:
								images_for_r1 = images.clone().detach().requires_grad_(True)
								real_pred_for_r1 = discriminator(images_for_r1)
								d_loss, d_loss_dict = loss_fn.compute_discriminator_loss(
									real_pred_for_r1, fake_pred, images_for_r1, apply_r1=True
								)
							else:
								d_loss, d_loss_dict = loss_fn.compute_discriminator_loss(
									real_pred, fake_pred, apply_r1=False
								)
						
						accelerator.backward(d_loss)
						
						if accelerator.sync_gradients and args.max_grad_norm:
							accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
						
						optim_d.step()

				# =========================
				# LOGGING AND CHECKPOINTING
				# =========================
				if accelerator.sync_gradients:
					if not (global_step % args.save_every) and accelerator.is_main_process:
						save_ckpt(accelerator,
								  model,
								  discriminator,
								  optim,
								  optim_d,
								  scheduler,
								  ema,
								  global_step,
								  os.path.join(args.ckpt_saved_dir, f'{wandb.run.name}-step-{global_step}.pth'))
					
					if not (global_step % args.sample_every):
						sample_prompts(
							model,
							val_dl,
							accelerator,
							device,
							global_step,
							use_ema=args.use_ema,
							ema=ema
						)
					
					# Prepare logging
					log_dict = {
						"g_loss": g_loss.item(),
						"lr": optim.param_groups[0]['lr']
					}
					
					# Add generator loss components
					for key, value in g_loss_dict.items():
						log_dict[f"g_{key}"] = value
					
					# Add discriminator losses if training
					if global_step >= args.discriminator_start_step:
						log_dict["d_loss"] = d_loss.item()
						for key, value in d_loss_dict.items():
							log_dict[f"d_{key}"] = value
					
					accelerator.log(log_dict, step=global_step)
					global_step += 1

	accelerator.end_training()        
	print("Train finished!")


if __name__ == "__main__":
	
	import argparse
	parser = argparse.ArgumentParser()

	# project / dataset
	parser.add_argument('--project_name', type=str, default='TexTok')
	parser.add_argument('--root', type=str, default='/media/pranoy/Datasets/coco-dataset/coco',help="Path to dataset")
	parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
	parser.add_argument('--batch_size', type=int, default=1, help="Batch size per device")
	parser.add_argument('--num_workers', type=int, default=4, help="Number of data loader workers")

	# training hyperparameters
	parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
	parser.add_argument('--num_epochs', type=int, default=1, help="Number of training epochs")
	parser.add_argument('--warmup_steps', type=int, default=4000, help="LR warmup steps")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help="Gradient accumulation steps")
	parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm for clipping")
	parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'], help="Mixed precision training mode")

	# EMA
	parser.add_argument('--use_ema', type=bool, default=True, help="Use EMA for model weights")
	parser.add_argument('--ema_decay', type=float, default=0.999, help="EMA decay rate")
	parser.add_argument('--ema_update_after', type=int, default=100,
						help="Steps before starting EMA updates")

	# adversarial training
	parser.add_argument('--discriminator_start_step', type=int, default=5000,
						help="Step to start discriminator training")
	parser.add_argument('--r1_every', type=int, default=16,
						help="R1 regularization interval")

	# logging / checkpointing
	parser.add_argument('--ckpt_every', type=int, default=10000, help="Save checkpoint every N steps")
	parser.add_argument('--eval_every', type=int, default=1000, help="Evaluate every N steps")
	parser.add_argument('--save_every', type=int, default=10000, help="Save model every N steps")
	parser.add_argument('--sample_every', type=int, default=1000, help="Sample and log reconstructions every N steps")
	parser.add_argument('--ckpt_saved_dir', type=str, default='ckpt', help="Directory to save outputs")

	args = parser.parse_args()


	kwargs = vars(args)
	print("Training configuration:")
	for k, v in kwargs.items():
		print(f"  {k}: {v}")


	train(args)






