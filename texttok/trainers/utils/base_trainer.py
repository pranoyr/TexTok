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



class BaseTrainer(object):
	def __init__(
		self, 
		cfg,
		model,
		dataloaders
		):

		self.cfg = cfg
		self.project_name = cfg.experiment.project_name
		self.exp_name = cfg.experiment.exp_name
		
		# init accelerator
		self.accelerator = Accelerator(
			mixed_precision=cfg.training.mixed_precision,
			gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
			log_with="wandb"
		)
		self.accelerator.init_trackers(
				project_name=cfg.experiment.project_name,
				init_kwargs={"wandb": {
				"config" : cfg,
				"name" : self.exp_name}
		})

		# models and dataloaders
		self.model = model
		self.train_dl , self.val_dl = dataloaders
		self.global_step = 0
		self.num_epoch = cfg.training.num_epochs
		self.gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
		self.batch_size = cfg.dataset.params.batch_size
		self.max_grad_norm = cfg.training.max_grad_norm

		# logging details
		self.num_epoch = cfg.training.num_epochs
		self.save_every = cfg.experiment.save_every
		self.sample_every = cfg.experiment.sample_every
		self.log_every = cfg.experiment.log_every
		self.eval_every = cfg.experiment.eval_every
		
	
		# Checkpoint and generated images folder
		output_folder = f"outputs/{cfg.experiment.project_name}"
		self.checkpoint_folder = os.path.join(output_folder, 'checkpoints')
		os.makedirs(self.checkpoint_folder, exist_ok=True)
		
		self.image_saved_dir = os.path.join(output_folder, 'images')
		os.makedirs(self.image_saved_dir, exist_ok=True)


		logging.info(f"Train dataset size: {len(self.train_dl.dataset)}")
		logging.info(f"Val dataset size: {len(self.val_dl.dataset)}")


		self.num_training_steps = self.num_epoch * math.ceil(len(self.train_dl) / self.gradient_accumulation_steps)
		logging.info(f"Total training steps: {self.num_training_steps}")

		
	@property
	def device(self):
		return self.accelerator.device
	
	
	def train(self):
		raise NotImplementedError("Train method not implemented")
		

	def save_ckpt(self, rewrite=False):
		"""Save checkpoint"""

		filename = os.path.join(self.checkpoint_folder, f'{self.project_name}_{self.exp_name}_step_{self.global_step}.pt')
		if rewrite:
			filename = os.path.join(self.checkpoint_folder, f'{self.project_name}_{self.exp_name}.pt')
		
		checkpoint={
				'step': self.global_step,
				'g_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
				'd_state_dict': self.accelerator.unwrap_model(self.discriminator).state_dict(),
				'g_optimizer_state_dict':self.optim.state_dict(),
				'd_optimizer_state_dict':self.optim_d.state_dict(),
				'scheduler_state_dict': self.scheduler.state_dict(),
				'ema_state_dict': self.ema.ema_model.state_dict(),
				'config': self.cfg

			}

		self.accelerator.save(checkpoint, filename)
		logging.info("Saving checkpoint: %s ...", filename)
   
   
	def resume_from_checkpoint(self):
		"""Resume from checkpoint with proper error handling"""
		checkpoint_path = self.cfg.experiment.resume_path_from_checkpoint
		
		if checkpoint_path:
			checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
			self.global_step = checkpoint['step']
			self.model.load_state_dict(checkpoint['g_state_dict'])
			self.discriminator.load_state_dict(checkpoint['d_state_dict'])
			self.optim.load_state_dict(checkpoint['g_optimizer_state_dict'])
			self.optim_d.load_state_dict(checkpoint['d_optimizer_state_dict'])
			self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
			self.ema.ema_model.load_state_dict(checkpoint['ema_state_dict'])
			
			logging.info(f"Resumed from checkpoint: {checkpoint_path} at step {self.global_step}")
		



	@torch.no_grad()
	def evaluate(self):
		raise NotImplementedError("Evaluate method not implemented")


