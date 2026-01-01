"""Quick plotting runner for training results visualization."""
import sys
from pathlib import Path
from src.plotting import load_and_plot_training_curves, show_confusion_matrices


if __name__ == '__main__':
    runs_root = './workspace/runs'
    if len(sys.argv) > 1:
        runs_root = sys.argv[1]
    
    print(f"📊 Loading training results from {runs_root}...")
    load_and_plot_training_curves(runs_root, 'converted_run')
    show_confusion_matrices(runs_root, 'converted_run')
