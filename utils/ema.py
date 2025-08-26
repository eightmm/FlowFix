"""
Exponential Moving Average (EMA) implementation for model weights.
Similar to DiffDock's approach for stable inference.
"""

import torch
import torch.nn as nn
from typing import Optional
import copy


class EMA:
    """
    Exponential Moving Average for model parameters.
    Maintains shadow weights that are updated with decay.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (0.999 = slow update, 0.9 = fast update)
            device: Device for EMA model
        """
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device
        
        # Create shadow model
        self.shadow_model = copy.deepcopy(model)
        self.shadow_model.eval()
        
        # Disable gradient computation for shadow model
        for param in self.shadow_model.parameters():
            param.requires_grad = False
            
        self.num_updates = 0
    
    @torch.no_grad()
    def update(self):
        """
        Update EMA parameters.
        Should be called after each optimizer step.
        """
        self.num_updates += 1
        
        # Use lower decay at the beginning for faster initial updates
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        for ema_param, model_param in zip(
            self.shadow_model.parameters(),
            self.model.parameters()
        ):
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1 - decay)
    
    def apply_shadow(self):
        """
        Apply shadow weights to the main model.
        Useful for evaluation with EMA weights.
        """
        self.backup = {}
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data.clone()
            
        for model_param, shadow_param in zip(
            self.model.parameters(),
            self.shadow_model.parameters()
        ):
            model_param.data.copy_(shadow_param.data)
    
    def restore(self):
        """
        Restore original model weights after applying shadow.
        """
        for name, param in self.model.named_parameters():
            param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state for checkpointing."""
        return {
            'shadow_model': self.shadow_model.state_dict(),
            'decay': self.decay,
            'num_updates': self.num_updates
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state from checkpoint."""
        self.shadow_model.load_state_dict(state_dict['shadow_model'])
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']