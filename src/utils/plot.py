import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import pandas as pd

def plot_scatter(y_true, y_pred, save_dir, type='valid', logger=None):
    """
    Draw scatter plot of true vs predicted values with regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_dir: Directory to save the plot
        type: Type of data ('train' or 'valid')
        logger: Logger object for logging messages
    """
    plt.figure(figsize=(10, 10))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Data points')
    
    # Plot the perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    # Add metrics text box
    metrics_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}\nPearson: {pearson_corr:.4f}'
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             verticalalignment='top',
             fontsize=10)
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{type.capitalize()} Set: True vs Predicted Values')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Make plot square with equal axes
    plt.axis('square')
    
    # Add colorbar to show density
    from scipy.stats import gaussian_kde
    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = y_true[idx], y_pred[idx], z[idx]
    scatter = plt.scatter(x, y, c=z, s=50, alpha=0.5, cmap='viridis')
    plt.colorbar(scatter, label='Density')
    
    save_path = os.path.join(save_dir, f'{type}_scatter_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Scatter plot saved to {save_path}")
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Pearson': pearson_corr
    }

def plot_losses(train_loss, valid_loss, save_dir, best_epoch=None, logger=None):
    """
    Plot training and validation losses over epochs
    
    Args:
        train_loss: List of training loss values
        valid_loss: List of validation loss values
        save_dir: Directory to save the plot
        best_epoch: Best epoch index (0-based)
        logger: Logger object
    """
    if not train_loss or not valid_loss:
        if logger:
            logger.warning("No loss data to plot")
        return
    
    # Get epochs for x-axis (assuming train_loss and valid_loss have the same length)
    start_epoch = 1  # default start epoch
    epochs = list(range(start_epoch, start_epoch + len(train_loss)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, valid_loss, 'r-', label='Validation Loss')
    
    # Always show best_epoch if provided, but display all epochs regardless
    if best_epoch:
        plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best epoch')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_history.png'), dpi=300)
    plt.close()
    
    if logger:
        logger.info(f"Loss history plot saved to {save_dir}/loss_history.png")

def plot_metrics(train_metrics, valid_metrics, save_dir, best_epoch=None, logger=None):
    """Plot metrics history
    
    Args:
        train_metrics: List of training metrics dictionaries
        valid_metrics: List of validation metrics dictionaries
        save_dir: Directory to save plots
        best_epoch: Marker for best epoch (optional)
        logger: Logger for logging messages
    """
    if not train_metrics or not valid_metrics:
        if logger:
            logger.warning("No metrics to plot")
        return
    
    # Create figures directory
    figures_dir = os.path.join(save_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot regression metrics
    plot_regression_metrics(train_metrics, valid_metrics, figures_dir, best_epoch=best_epoch, logger=logger)
    
    if logger:
        logger.info(f"Saved metrics plots to {figures_dir}")

def plot_regression_metrics(train_metrics, valid_metrics, save_dir, best_epoch=None, logger=None):
    """Plot regression metrics
    
    Args:
        train_metrics: List of training metrics dictionaries
        valid_metrics: List of validation metrics dictionaries
        save_dir: Directory to save plots
        best_epoch: Marker for best epoch (optional)
        logger: Logger for logging messages
    """
    # Check if metrics are available
    if (
        len(valid_metrics) == 0 or
        'metrics' not in valid_metrics[0] or
        'regression' not in valid_metrics[0]['metrics']
    ):
        if logger:
            logger.warning("No regression metrics to plot")
        return
    
    # Extract regression metrics
    epochs = list(range(1, len(train_metrics) + 1))
    
    # Get all metrics (will be available in validation results)
    metric_names = list(valid_metrics[0]['metrics']['regression'].keys())
    
    # Create plot for each metric
    for metric_name in metric_names:
        try:
            valid_values = [valid_metrics[i]['metrics']['regression'][metric_name] for i in range(len(valid_metrics))]
            
            plt.figure(figsize=(10, 6))
            
            # Plot only validation values (training may not have all metrics)
            plt.plot(epochs, valid_values, 'b-', label=f'Validation {metric_name}')
            
            if best_epoch:
                # Mark best epoch
                plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
                
                # Get best value
                best_value = valid_values[best_epoch - 1] if 0 <= best_epoch - 1 < len(valid_values) else None
                if best_value is not None:
                    plt.scatter([best_epoch], [best_value], color='r', s=100, zorder=5)
                    plt.text(best_epoch, best_value, f' {best_value:.4f}', verticalalignment='bottom')
            
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'Regression {metric_name} vs Epoch')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'regression_{metric_name.lower()}.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            if logger:
                logger.warning(f"Error plotting {metric_name}: {str(e)}")
            continue
    
    if logger:
        logger.info(f"Saved regression metrics plots to {save_dir}")

def plot_confusion_matrix(cm, classes, save_dir, title='Confusion Matrix', task_type=None, cmap=plt.cm.Blues):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        classes: List of class names
        save_dir: Directory to save the plot
        title: Plot title
        task_type: Task type for filename (optional)
        cmap: Colormap
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # 파일명 설정 (task_type이 있으면 사용, 없으면 기본 파일명)
    filename = f"{task_type}_cm.png" if task_type else "confusion_matrix.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def plot_binding_metrics(y_true, y_pred, save_dir, task_type='binding', epoch=None, logger=None):
    """
    Plot metrics for binding or non-binding classification tasks in a single figure
    with 3 subplots: confusion matrix, ROC curve, and PR curve.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        save_dir: Directory to save the plot
        task_type: 'binding' or 'non_binding'
        epoch: Current epoch number (not used in filename anymore)
        logger: Logger object
    """
    # Convert predictions to binary (0/1) using 0.5 threshold
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle(f'{task_type.replace("_", " ").title()} Classification Metrics', fontsize=16)
    
    # 1. Plot confusion matrix
    axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('Confusion Matrix')
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Negative', 'Positive'])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Negative', 'Positive'])
    axes[0].set_ylabel('True label')
    axes[0].set_xlabel('Predicted label')
    
    # 2. Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Plot precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    
    axes[2].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0.0, 1.05])
    
    plt.tight_layout()

    # Save the combined figure without epoch number (already implemented)
    save_path = os.path.join(save_dir, f'{task_type}_metrics.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    if logger:
        logger.info(f"{task_type.replace('_', ' ').title()} metrics plot saved to {save_path}")

def plot_loss_types(train_metrics, valid_metrics, save_dir, best_epoch=None, logger=None):
    """
    Plot different types of losses over training epochs in a 2x2 subplot format
    """
    # Always start from the first epoch
    if 'epoch' in train_metrics[0]:
        start_epoch = train_metrics[0]['epoch']
    else:
        start_epoch = 1
    
    epochs = list(range(start_epoch, start_epoch + len(train_metrics)))
    
    # Plot loss types (reg_loss, bind_loss, nonbind_loss, total_loss)
    plt.figure(figsize=(16, 12))
    plt.suptitle('Loss Types Over Training', fontsize=16)
    
    # Total Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [m['total_loss'] for m in train_metrics], 'b-', label='Training')
    plt.plot(epochs, [m['losses']['total_loss'] for m in valid_metrics], 'r-', label='Validation')
    if best_epoch:
        plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    plt.grid(True)
    
    # Regression Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m['reg_loss'] for m in train_metrics], 'b-', label='Training')
    plt.plot(epochs, [m['losses']['reg_loss'] for m in valid_metrics], 'r-', label='Validation')
    if best_epoch:
        plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Regression Loss')
    plt.legend()
    plt.grid(True)
    
    # Binding Loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['bind_loss'] for m in train_metrics], 'b-', label='Training')
    plt.plot(epochs, [m['losses']['bind_loss'] for m in valid_metrics], 'r-', label='Validation')
    if best_epoch:
        plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Binding Loss')
    plt.legend()
    plt.grid(True)
    
    # Non-binding Loss
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [m['nonbind_loss'] for m in train_metrics], 'b-', label='Training')
    plt.plot(epochs, [m['losses']['nonbind_loss'] for m in valid_metrics], 'r-', label='Validation')
    if best_epoch:
        plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Non-binding Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_types.png'), dpi=300)
    plt.close()
    
    if logger:
        logger.info(f"Loss types plot saved to {save_dir}/loss_types.png")

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_loss_curves(train_losses, valid_losses, save_dir, title="Training History"):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        valid_losses: List of validation losses
        save_dir: Directory to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, valid_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Find best epoch (minimum validation loss)
    best_epoch = np.argmin(valid_losses) + 1
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
    plt.legend()
    
    # Loss difference subplot
    plt.subplot(1, 2, 2)
    loss_diff = np.array(valid_losses) - np.array(train_losses)
    plt.plot(epochs, loss_diff, 'purple', linewidth=2)
    plt.title('Validation - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to {save_dir}/loss_curves.png")

def plot_r2_curves(train_r2_scores, valid_r2_scores, save_dir, title="R² Score History"):
    """
    Plot training and validation R² score curves
    
    Args:
        train_r2_scores: List of training R² scores
        valid_r2_scores: List of validation R² scores
        save_dir: Directory to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 5))
    
    # R² subplot
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_r2_scores) + 1)
    
    plt.plot(epochs, train_r2_scores, 'b-', label='Training R²', linewidth=2)
    plt.plot(epochs, valid_r2_scores, 'r-', label='Validation R²', linewidth=2)
    
    plt.title('R² Score Curves')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Find best epoch (maximum validation R²)
    best_epoch = np.argmax(valid_r2_scores) + 1
    best_r2 = valid_r2_scores[best_epoch - 1]
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
    plt.scatter([best_epoch], [best_r2], color='g', s=100, zorder=5)
    plt.text(best_epoch, best_r2, f' {best_r2:.4f}', verticalalignment='bottom', fontweight='bold')
    plt.legend()
    
    # R² difference subplot
    plt.subplot(1, 2, 2)
    r2_diff = np.array(train_r2_scores) - np.array(valid_r2_scores)
    plt.plot(epochs, r2_diff, 'purple', linewidth=2)
    plt.title('Training - Validation R²')
    plt.xlabel('Epoch')
    plt.ylabel('R² Difference')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add interpretation guidelines
    plt.text(0.02, 0.98, 'Positive: Overfitting\nNegative: Underfitting', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'r2_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"R² curves saved to {save_dir}/r2_curves.png")
    
    # Print summary statistics
    print(f"Best validation R²: {best_r2:.4f} at epoch {best_epoch}")
    print(f"Final validation R²: {valid_r2_scores[-1]:.4f}")
    print(f"Max training R²: {max(train_r2_scores):.4f}")
    print(f"Max validation R²: {max(valid_r2_scores):.4f}")

def plot_predictions(true_values, predictions, save_dir, title="Prediction Results"):
    """Plot prediction vs true values scatter plot"""
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Convert to numpy arrays
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    
    # Calculate metrics
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predictions)
    pearson_r, pearson_p = pearsonr(true_values, predictions)
    
    # 1. Scatter plot
    ax1.scatter(true_values, predictions, alpha=0.6, s=30)
    
    # Perfect prediction line
    min_val = min(true_values.min(), predictions.min())
    max_val = max(true_values.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax1.set_xlabel('True Affinity', fontsize=12)
    ax1.set_ylabel('Predicted Affinity', fontsize=12)
    ax1.set_title(f'Predictions vs True Values\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual plot
    residuals = predictions - true_values
    ax2.scatter(predictions, residuals, alpha=0.6, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Affinity', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title(f'Residual Plot\nMAE = {mae:.4f}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution of residuals
    ax3.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(residuals.mean(), color='r', linestyle='--', lw=2, 
                label=f'Mean: {residuals.mean():.4f}')
    ax3.axvline(0, color='g', linestyle='-', lw=2, label='Zero')
    ax3.set_xlabel('Residuals', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Distribution of Residuals', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    abs_errors = np.abs(residuals)
    ax4.hist(abs_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax4.axvline(abs_errors.mean(), color='r', linestyle='--', lw=2, 
                label=f'Mean AE: {abs_errors.mean():.4f}')
    ax4.axvline(np.median(abs_errors), color='g', linestyle='--', lw=2, 
                label=f'Median AE: {np.median(abs_errors):.4f}')
    ax4.set_xlabel('Absolute Error', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Absolute Errors', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add overall metrics text
    metrics_text = f"""
    Metrics Summary:
    ├─ RMSE: {rmse:.4f}
    ├─ MAE: {mae:.4f}
    ├─ R²: {r2:.4f}
    ├─ Pearson R: {pearson_r:.4f}
    └─ Pearson P: {pearson_p:.4e}
    """
    
    fig.text(0.02, 0.02, metrics_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.15)
    plt.savefig(os.path.join(save_dir, 'prediction_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_analysis(true_values, predictions, save_dir):
    """Plot detailed error analysis"""
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    residuals = predictions - true_values
    abs_errors = np.abs(residuals)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Error vs True Values
    ax1.scatter(true_values, abs_errors, alpha=0.6, s=30)
    ax1.set_xlabel('True Affinity', fontsize=12)
    ax1.set_ylabel('Absolute Error', fontsize=12)
    ax1.set_title('Absolute Error vs True Values', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(true_values, abs_errors, 1)
    p = np.poly1d(z)
    ax1.plot(sorted(true_values), p(sorted(true_values)), "r--", alpha=0.8)
    
    # 2. Error vs Predicted Values
    ax2.scatter(predictions, abs_errors, alpha=0.6, s=30, color='orange')
    ax2.set_xlabel('Predicted Affinity', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('Absolute Error vs Predictions', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q Plot for residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot of Residuals', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative error distribution
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax4.plot(sorted_errors, cumulative, linewidth=2)
    ax4.set_xlabel('Absolute Error', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('Cumulative Error Distribution', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # Add percentile lines
    percentiles = [0.5, 0.8, 0.9, 0.95]
    for p in percentiles:
        error_at_p = np.percentile(abs_errors, p * 100)
        ax4.axvline(error_at_p, color='red', linestyle='--', alpha=0.7)
        ax4.text(error_at_p, p, f'{p*100:.0f}%', rotation=90, 
                verticalalignment='bottom', fontsize=10)
    
    plt.suptitle('Error Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_affinity_distribution(true_values, predictions, save_dir):
    """Plot distribution comparison of true vs predicted affinities"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. True values distribution
    ax1.hist(true_values, bins=50, alpha=0.7, label='True Affinity', color='blue', edgecolor='black')
    ax1.set_xlabel('Affinity', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('True Affinity Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Predicted values distribution
    ax2.hist(predictions, bins=50, alpha=0.7, label='Predicted Affinity', color='red', edgecolor='black')
    ax2.set_xlabel('Affinity', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Predicted Affinity Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Overlapped distributions
    ax3.hist(true_values, bins=50, alpha=0.5, label='True Affinity', color='blue', density=True)
    ax3.hist(predictions, bins=50, alpha=0.5, label='Predicted Affinity', color='red', density=True)
    ax3.set_xlabel('Affinity', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Distribution Comparison', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'affinity_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(true_values, predictions, save_dir, model_info=None):
    """Create a comprehensive summary report"""
    true_values = np.array(true_values)
    predictions = np.array(predictions)
    
    # Calculate all metrics
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predictions)
    pearson_r, pearson_p = pearsonr(true_values, predictions)
    
    residuals = predictions - true_values
    abs_errors = np.abs(residuals)
    
    # Create summary text
    report = f"""
# Model Performance Report

## Model Information
{f"- Model: {model_info.get('model_name', 'EGNN Affinity Model')}" if model_info else "- Model: EGNN Affinity Model"}
{f"- Parameters: {model_info.get('total_parameters', 'N/A')}" if model_info else ""}
{f"- Training Time: {model_info.get('training_time', 'N/A')}" if model_info else ""}

## Dataset Statistics
- Total Samples: {len(true_values):,}
- True Affinity Range: [{true_values.min():.4f}, {true_values.max():.4f}]
- True Affinity Mean ± Std: {true_values.mean():.4f} ± {true_values.std():.4f}
- Predicted Affinity Range: [{predictions.min():.4f}, {predictions.max():.4f}]
- Predicted Affinity Mean ± Std: {predictions.mean():.4f} ± {predictions.std():.4f}

## Performance Metrics
### Primary Metrics
- Root Mean Square Error (RMSE): {rmse:.4f}
- Mean Absolute Error (MAE): {mae:.4f}
- R-squared Score (R²): {r2:.4f}
- Pearson Correlation: {pearson_r:.4f} (p-value: {pearson_p:.2e})

### Error Statistics
- Mean Residual: {residuals.mean():.4f}
- Std Residual: {residuals.std():.4f}
- Mean Absolute Error: {abs_errors.mean():.4f}
- Median Absolute Error: {np.median(abs_errors):.4f}
- 90th Percentile Error: {np.percentile(abs_errors, 90):.4f}
- 95th Percentile Error: {np.percentile(abs_errors, 95):.4f}
- Max Absolute Error: {abs_errors.max():.4f}

### Error Distribution
- Errors < 0.5: {(abs_errors < 0.5).sum():,} ({(abs_errors < 0.5).mean()*100:.1f}%)
- Errors < 1.0: {(abs_errors < 1.0).sum():,} ({(abs_errors < 1.0).mean()*100:.1f}%)
- Errors < 2.0: {(abs_errors < 2.0).sum():,} ({(abs_errors < 2.0).mean()*100:.1f}%)
- Errors ≥ 2.0: {(abs_errors >= 2.0).sum():,} ({(abs_errors >= 2.0).mean()*100:.1f}%)

## Summary
The model achieves {'excellent' if r2 > 0.8 else 'good' if r2 > 0.6 else 'moderate' if r2 > 0.4 else 'poor'} 
performance with an R² score of {r2:.4f} and RMSE of {rmse:.4f}.
"""
    
    # Save report
    with open(os.path.join(save_dir, 'performance_report.md'), 'w') as f:
        f.write(report)
    
    # Also create a CSV with detailed results
    results_df = pd.DataFrame({
        'True_Affinity': true_values,
        'Predicted_Affinity': predictions,
        'Residual': residuals,
        'Absolute_Error': abs_errors
    })
    
    results_df.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)
    
    print("Summary report and detailed results saved.")

def plot_all_results(true_values, predictions, save_dir, model_info=None):
    """Generate all plots and reports"""
    print("Generating comprehensive result analysis...")
    
    # Generate all plots
    plot_predictions(true_values, predictions, save_dir)
    plot_error_analysis(true_values, predictions, save_dir)
    plot_affinity_distribution(true_values, predictions, save_dir)
    
    # Create summary report
    create_summary_report(true_values, predictions, save_dir, model_info)
    
    print(f"All results saved to: {save_dir}")