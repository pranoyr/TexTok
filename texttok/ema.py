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
        
        # Set device once during initialization
        device = next(model.parameters()).device
        self.ema_model = self.ema_model.to(device)
        self.device = device  # Cache device for efficiency
        
        # Freeze EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
        print(f"EMA initialized with decay={decay}, update_after={update_after} on device={device}")
    
    def update(self, model: nn.Module):
        """Update EMA weights after each training step"""
        self.step += 1
        
        if self.step <= self.update_after:
            self.copy_params_from_model(model)
            return
        
        # EMA update - no device checking needed since both should be on same device
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                # More efficient EMA update using in-place operations
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def copy_params_from_model(self, model: nn.Module):
        """Copy all parameters from model to EMA model during warmup"""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.copy_(model_param.data)
    
    def to(self, device):
        """Move EMA model to device if needed"""
        if device != self.device:
            self.ema_model = self.ema_model.to(device)
            self.device = device
        return self