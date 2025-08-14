import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, update_after: int = 100):
        self.decay = decay
        self.update_after = update_after
        self.step = 0
        
        # Create EMA copy of the model
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        
        # 🔥 FIX: Ensure EMA model is on same device as original model
        self.ema_model = self.ema_model.to(next(model.parameters()).device)
        
        # Freeze EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
        print(f"EMA initialized with decay={decay}, update_after={update_after}")
    
    def update(self, model: nn.Module):
        """Update EMA weights after each training step"""
        self.step += 1
        
        if self.step <= self.update_after:
            self.copy_params_from_model(model)
            return
        
        # 🔥 FIX: Ensure both models are on same device during update
        model_device = next(model.parameters()).device
        ema_device = next(self.ema_model.parameters()).device
        
        if model_device != ema_device:
            self.ema_model = self.ema_model.to(model_device)
        
        # EMA update
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data = self.decay * ema_param.data + (1 - self.decay) * model_param.data
    
    def copy_params_from_model(self, model: nn.Module):
        """Copy all parameters from model to EMA model"""
        # 🔥 FIX: Ensure same device during copy
        model_device = next(model.parameters()).device
        self.ema_model = self.ema_model.to(model_device)
        
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.copy_(model_param.data)