"""Plotting utilities for training results visualization."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob


def load_and_plot_training_curves(runs_root: str, run_name: str = 'converted_run'):
    """Load ultralytics results.csv and plot accuracy/loss curves.
    
    Args:
        runs_root: Path to ultralytics runs folder (e.g., workspace/runs)
        run_name: Name of the run to plot (e.g., 'converted_run')
    """
    runs_root = Path(runs_root)
    
    # Find the run folder
    candidates = list(runs_root.glob(f'classify/{run_name}*/results.csv'))
    if not candidates:
        print(f"No results.csv found under {runs_root}/classify/{run_name}*/")
        print("Available runs:", list(runs_root.glob('classify/*/results.csv')))
        return
    
    results_file = candidates[0]
    print(f"Loading results from {results_file}")
    df = pd.read_csv(results_file)
    df.columns = [c.strip() for c in df.columns]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy
    ax = axes[0, 0]
    acc_cols = [c for c in df.columns if 'top1' in c.lower() or ('acc' in c.lower() and 'val' not in c.lower())]
    if acc_cols:
        ax.plot(df['epoch'], df[acc_cols[0]], label='Train', marker='o')
    val_acc_cols = [c for c in df.columns if 'top1' in c.lower() and 'val' in c.lower()]
    if val_acc_cols:
        ax.plot(df['epoch'], df[val_acc_cols[0]], label='Val', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss
    ax = axes[0, 1]
    loss_cols = [c for c in df.columns if 'loss' in c.lower() and 'val' not in c.lower()]
    if loss_cols:
        ax.plot(df['epoch'], df[loss_cols[0]], label='Train Loss', marker='o', color='red')
    val_loss_cols = [c for c in df.columns if 'loss' in c.lower() and 'val' in c.lower()]
    if val_loss_cols:
        ax.plot(df['epoch'], df[val_loss_cols[0]], label='Val Loss', marker='s', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss per Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: All columns (raw data)
    ax = axes[1, 0]
    for col in df.columns:
        if col != 'epoch':
            try:
                ax.plot(df['epoch'], df[col], label=col, alpha=0.7)
            except:
                pass
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('All Metrics')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary stats
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    Training Summary
    ================
    Run: {run_name}
    Epochs: {len(df)}
    
    Final Epoch Metrics:
    """
    last_row = df.iloc[-1]
    for col in df.columns:
        if col != 'epoch':
            try:
                val = last_row[col]
                if isinstance(val, float):
                    summary_text += f"\n{col}: {val:.4f}"
            except:
                pass
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    plt.savefig(str(runs_root / f'{run_name}_curves.png'), dpi=100, bbox_inches='tight')
    print(f"Saved plot to {runs_root / f'{run_name}_curves.png'}")
    plt.show()


def show_confusion_matrices(runs_root: str, run_name: str = 'converted_run'):
    """Display confusion matrix images from ultralytics run.
    
    Args:
        runs_root: Path to ultralytics runs folder
        run_name: Name of the run
    """
    runs_root = Path(runs_root)
    
    # Find confusion matrix files
    candidates = list(runs_root.glob(f'classify/{run_name}*/confusion_matrix*.png'))
    if not candidates:
        print(f"No confusion_matrix*.png found under {runs_root}/classify/{run_name}*/")
        return
    
    from PIL import Image
    for cm_path in candidates:
        img = Image.open(cm_path)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Confusion Matrix: {cm_path.parent.name}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Example usage
    import sys
    if len(sys.argv) > 1:
        runs_root = sys.argv[1]
    else:
        runs_root = './workspace/runs'
    
    load_and_plot_training_curves(runs_root, 'converted_run')
    show_confusion_matrices(runs_root, 'converted_run')
