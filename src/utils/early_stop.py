import torch
from datetime import datetime
import os

class EarlyStopping(object):
    def __init__(self, mode='min', patience=50, min_delta=0.0, restore_best_weights=True, filename=None, save_dir='./save'):
        """
        Early stopping with flexible mode and min_delta support
        
        Args:
            mode: 'min' to minimize metric, 'max' to maximize metric
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
            filename: Custom filename for saving checkpoints
            save_dir: Directory to save checkpoints
        """
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        if filename is None:
            dt = datetime.now()
            filename = os.path.join(save_dir, f'early_stop_{dt.strftime("%Y-%m-%d_%H-%M-%S")}.pth')

        assert mode in ['min', 'max', 'lower', 'higher']  # Support both old and new modes
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        # Map old modes to new ones for backward compatibility
        if mode == 'lower':
            self.mode = 'min'
        elif mode == 'higher':
            self.mode = 'max'
            
        if self.mode == 'max':
            self._check = self._check_max
            self.best_score = -float('inf')
        else:
            self._check = self._check_min
            self.best_score = float('inf')

        self.patience = patience
        self.counter = 0
        self.timestep = 0
        self.filename = filename
        self.save_dir = save_dir
        self.early_stop = False
        
        # Store model state for restoration
        self.best_model_state = None
        
        self.training_history = {
            'best_score': None,
            'best_epoch': None,
            'scores': [],
            'epochs': []
        }

    def _check_max(self, score, prev_best_score):
        """Check if score is better for maximization (considering min_delta)"""
        return score > prev_best_score + self.min_delta

    def _check_min(self, score, prev_best_score):
        """Check if score is better for minimization (considering min_delta)"""
        return score < prev_best_score - self.min_delta

    def __call__(self, score, model, optimizer=None, scheduler=None, epoch=None, **kwargs):
        """
        Callable interface for compatibility with run.py
        
        Args:
            score: Current validation score
            model: PyTorch model
            
        Returns:
            bool: True if early stopping should be triggered
        """
        return self.step(score, model, optimizer, scheduler, epoch, **kwargs)

    def step(self, score, model, optimizer=None, scheduler=None, epoch=None, **kwargs):
        """
        Early stopping step with extended state saving
        
        Args:
            score: Current validation score
            model: PyTorch model
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            epoch: Current epoch number
            **kwargs: Additional state to save
        """
        self.timestep += 1
        self.training_history['scores'].append(score)
        self.training_history['epochs'].append(epoch if epoch is not None else self.timestep)
        
        # Initialize best score on first call
        if self.training_history['best_score'] is None:
            self.best_score = score
            self.training_history['best_score'] = score
            self.training_history['best_epoch'] = epoch if epoch is not None else self.timestep
            if self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()
            self.save_checkpoint(model, optimizer, scheduler, epoch, **kwargs)
            
        elif self._check(score, self.best_score):
            # Found better score
            self.best_score = score
            self.training_history['best_score'] = score
            self.training_history['best_epoch'] = epoch if epoch is not None else self.timestep
            if self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()
            self.save_checkpoint(model, optimizer, scheduler, epoch, **kwargs)
            self.counter = 0
            
        else:
            # No improvement
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
                # Restore best weights if requested
                if self.restore_best_weights and self.best_model_state is not None:
                    print(f"Restoring best model weights from epoch {self.training_history['best_epoch']}")
                    model.load_state_dict(self.best_model_state)
                    
        return self.early_stop

    def save_checkpoint(self, model, optimizer=None, scheduler=None, epoch=None, train_metrics=None, valid_metrics=None, **kwargs):
        """
        Save checkpoint with extended state
        
        Args:
            model: PyTorch model
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            epoch: Current epoch number
            train_metrics: Training metrics history
            valid_metrics: Validation metrics history
            **kwargs: Additional state to save
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'timestep': self.timestep,
            'best_score': self.best_score,
            'counter': self.counter,
            'training_history': self.training_history,
            'epoch': epoch if epoch is not None else self.timestep,
            'mode': self.mode,
            'min_delta': self.min_delta
        }
        
        # Save training metrics
        if train_metrics is not None:
            checkpoint['train_metrics'] = train_metrics
        if valid_metrics is not None:
            checkpoint['valid_metrics'] = valid_metrics
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        # Add any additional state from kwargs
        checkpoint.update(kwargs)
        
        # Ensure the save directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        # Save temporary checkpoint first
        temp_filename = self.filename + '.tmp'
        torch.save(checkpoint, temp_filename)
        
        # If save is successful, rename to final filename
        if os.path.exists(self.filename):
            os.remove(self.filename)
        os.rename(temp_filename, self.filename)

    def load_checkpoint(self, model, optimizer=None, scheduler=None):
        """
        Load checkpoint with extended state
        
        Args:
            model: PyTorch model
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            
        Returns:
            dict: Loaded checkpoint state
        """
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"No checkpoint found at {self.filename}")
            
        checkpoint = torch.load(self.filename, weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Restore early stopping state
        self.timestep = checkpoint.get('timestep', 0)
        self.best_score = checkpoint.get('best_score', None)
        self.counter = checkpoint.get('counter', 0)
        self.mode = checkpoint.get('mode', 'min')
        self.min_delta = checkpoint.get('min_delta', 0.0)
        self.training_history = checkpoint.get('training_history', {
            'best_score': None,
            'best_epoch': None,
            'scores': [],
            'epochs': []
        })
        
        return checkpoint

    def get_best_score(self):
        """Get the best score achieved"""
        return self.training_history['best_score']
    
    def get_best_epoch(self):
        """Get the epoch with the best score"""
        return self.training_history['best_epoch']
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.timestep = 0
        self.early_stop = False
        self.best_model_state = None
        
        if self.mode == 'max':
            self.best_score = -float('inf')
        else:
            self.best_score = float('inf')
            
        self.training_history = {
            'best_score': None,
            'best_epoch': None,
            'scores': [],
            'epochs': []
        }