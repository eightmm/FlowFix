"""
Model initialization utilities for FlowFix.
"""

import torch
import torch.nn as nn
import math


def init_weights(module):
    """
    Initialize model weights for better training stability.
    """
    if isinstance(module, nn.Linear):
        # Xavier/Glorot initialization for linear layers
        torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)
    
    elif isinstance(module, nn.LayerNorm):
        # Standard initialization for LayerNorm
        if module.weight is not None:
            torch.nn.init.constant_(module.weight, 1.0)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)
    
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        # Standard initialization for BatchNorm
        if module.weight is not None:
            torch.nn.init.constant_(module.weight, 1.0)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)
    
    elif hasattr(module, 'weight') and hasattr(module.weight, 'data'):
        # For custom layers with weight parameters
        if module.weight.dim() > 1:
            # Initialize as matrices
            fan_in = module.weight.shape[1] if module.weight.dim() > 1 else 1
            fan_out = module.weight.shape[0]
            std = math.sqrt(2.0 / (fan_in + fan_out))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        else:
            # Initialize as vectors
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)
        
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)


def init_model(model):
    """
    Initialize all weights in a model.
    
    Args:
        model: PyTorch model to initialize
    
    Returns:
        Initialized model
    """
    # Apply initialization to all modules
    model.apply(init_weights)
    
    # Special handling for output layers - smaller initialization
    # to start with smaller predictions
    for name, module in model.named_modules():
        if 'output' in name.lower() or 'out_proj' in name.lower():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
    
    # Initialize learnable scale parameters if they exist
    for name, param in model.named_parameters():
        if 'scale' in name.lower() and param.dim() == 0:
            torch.nn.init.constant_(param, 1.0)
    
    return model