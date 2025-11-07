import dgl, torch, scipy
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from scipy.stats import spearmanr


def custom_loss(pred, true, direction='over', penalty_factor=2.0):
    base_loss = (pred - true) ** 2
    if direction == 'over':
        penalty = torch.where(pred > true, penalty_factor * base_loss, base_loss)
    elif direction == 'under':
        penalty = torch.where(pred < true, penalty_factor * base_loss, base_loss)
    else:
        penalty = base_loss
        
    return torch.mean(penalty)
    


def run_train_epoch(model, loader, optimizer, scheduler, device='cpu'):
    """Run a single training epoch.
    
    Args:
        model: The model to train
        loader: DataLoader providing batches
        optimizer: Optimizer for updating weights
        scheduler: Learning rate scheduler
        device: Device to use for computation
        
    Returns:
        Dictionary containing average losses and metrics
    """
    model.train()

    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training", total=len(loader)):
        prot_data, ligand_graph, ligand_mask, interaction_mask, reg_true, bind_true, nonbind_true = batch
        
        # Move data to device
        for key in prot_data:
            prot_data[key] = prot_data[key].to(device)
        
        ligand_graph = ligand_graph.to(device)
        ligand_mask = ligand_mask.to(device)
        interaction_mask = interaction_mask.to(device)
        
        reg_true = reg_true.to(device)
        
        # Forward pass - regression only
        reg_pred = model(prot_data, ligand_graph, interaction_mask)
        
        # Calculate loss using custom loss function
#loss = custom_loss(reg_pred.squeeze(), reg_true, direction='over', penalty_factor=2.0)
        # Huber
        loss = F.huber_loss(reg_pred.squeeze(), reg_true)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

        torch.cuda.empty_cache()
    
    scheduler.step()
    
    return {
        'total_loss': total_loss / max(1, num_batches)
    }

@torch.no_grad()
def run_eval_epoch(model, loader, device='cpu'):
    """Run a single evaluation epoch.
    
    Args:
        model: The model to evaluate
        loader: DataLoader providing batches
        device: Device to use for computation
        
    Returns:
        Dictionary containing losses and metrics for regression task
    """
    model.eval()

    all_reg_true = []
    all_reg_pred = []
    
    total_loss = 0
    num_batches = 0

    for batch in tqdm(loader, desc="Evaluation", total=len(loader)):
        prot_data, ligand_graph, ligand_mask, interaction_mask, reg_true, bind_true, nonbind_true = batch
        
        # Move data to device
        for key in prot_data:
            prot_data[key] = prot_data[key].to(device)
        
        ligand_graph = ligand_graph.to(device)
        ligand_mask = ligand_mask.to(device)
        interaction_mask = interaction_mask.to(device)
        
        reg_true = reg_true.to(device)
        
        # Forward pass - regression only
        reg_pred = model(prot_data, ligand_graph, interaction_mask)
        
        # Calculate loss using custom loss function
#loss = custom_loss(reg_pred.squeeze(), reg_true, direction='over', penalty_factor=2.0)
        loss = F.huber_loss(reg_pred.squeeze(), reg_true)
        
        # Accumulate losses
        total_loss += loss.item()
        num_batches += 1
        
        # Collect predictions and true values
        all_reg_true.append(reg_true)
        all_reg_pred.append(reg_pred)
        
        torch.cuda.empty_cache()

    # Concatenate all predictions and true values
    reg_true = torch.cat(all_reg_true, dim=0)
    reg_pred = torch.cat(all_reg_pred, dim=0)
    
    # Calculate average loss
    avg_loss = total_loss / max(1, num_batches)
    
    # Compute metrics for regression task
    reg_metrics = compute_regression_metrics(reg_true, reg_pred)
    
    return {
        'losses': {
            'total_loss': avg_loss
        },
        'metrics': {
            'regression': reg_metrics
        }
    }

    
@torch.no_grad()
def compute_regression_metrics(true, pred):
    """Compute comprehensive regression metrics"""
    true = true.cpu().numpy()
    pred = pred.cpu().numpy().squeeze()
    
    # Basic error metrics
    mse = np.mean((true - pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true - pred))
    
    # Correlation metrics
    pearson = np.corrcoef(true, pred)[0, 1] if len(true) > 1 else 0
    spearman = spearmanr(true, pred)[0] if len(true) > 1 else 0
    
    # Explained variance metrics
    ss_tot = np.sum((true - true.mean()) ** 2)
    ss_res = np.sum((true - pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    residuals = true - pred
    mean_bias = np.mean(residuals)  # Bias (systematic error)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Pearson': pearson,
        'Spearman': spearman,
        'R2': r2,
        'Mean_Bias': mean_bias
    }
@torch.no_grad()
def compute_binary_metrics(true, pred):
    """Compute essential binary classification metrics"""
    true = true.cpu().numpy()
    pred_probs = torch.sigmoid(pred).cpu().numpy().squeeze()
    pred_class = (pred_probs >= 0.5).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(true, pred_class).ravel()
    
    return {
        'accuracy': accuracy_score(true, pred_class),
        'precision': precision_score(true, pred_class, zero_division=0),
        'recall': recall_score(true, pred_class, zero_division=0),
        'f1': f1_score(true, pred_class, zero_division=0),
        'auc_roc': roc_auc_score(true, pred_probs) if len(np.unique(true)) > 1 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0,
        'mcc': matthews_corrcoef(true, pred_class),
        'f1_macro': f1_score(true, pred_class, average='macro', zero_division=0),
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0
    }
    
