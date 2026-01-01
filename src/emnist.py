"""EMNIST dataset download, conversion to YOLO format, and benchmarking.
Ported from notebook Cell 8-9.
"""
import os
import shutil
import torch
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
from torchvision import datasets
from ultralytics import YOLO, settings


def download_and_prepare_emnist(emnist_dir: str, split: str = 'digits'):
    """Download EMNIST dataset and convert to YOLO classification format.
    
    Args:
        emnist_dir: Root directory to save EMNIST data
        split: EMNIST split (e.g., 'digits', 'letters', 'balanced')
    """
    emnist_dir = Path(emnist_dir)
    emnist_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading EMNIST ({split})...")
    training_data = datasets.EMNIST(root=str(emnist_dir / 'raw'), split=split, train=True, download=True)
    test_data = datasets.EMNIST(root=str(emnist_dir / 'raw'), split=split, train=False, download=True)
    
    def save_dataset_to_yolo(dataset, split_name):
        """Convert EMNIST dataset to YOLO classification folder structure."""
        save_path = emnist_dir / split_name
        if save_path.exists():
            print(f"✅ {split_name} set already exists. Skipping.")
            return
        
        print(f"⚙️  Converting {split_name} set to YOLO format...")
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create class folders (0-9 for digits, 0-25 for letters)
        num_classes = len(set([label for _, label in dataset]))
        for i in range(num_classes):
            (save_path / str(i)).mkdir(exist_ok=True)
        
        # Save images
        for idx, (image, label) in enumerate(tqdm(dataset, total=len(dataset))):
            # EMNIST images are rotated/flipped; transpose to fix
            img_array = np.array(image)
            img_array = np.transpose(img_array)
            
            filename = f"{split_name}_{idx}.jpg"
            cv2.imwrite(str(save_path / str(label) / filename), img_array)
    
    save_dataset_to_yolo(training_data, 'train')
    save_dataset_to_yolo(test_data, 'val')
    
    print(f"✅ EMNIST dataset ready at {emnist_dir}")
    return emnist_dir


def train_on_emnist(emnist_dir: str, work_dir: str, epochs: int = 25, batch: int = 64):
    """Train YOLO on EMNIST benchmark dataset.
    
    Args:
        emnist_dir: Path to prepared EMNIST dataset
        work_dir: Workspace directory for checkpoints/runs
        epochs: Number of epochs
        batch: Batch size
    
    Returns:
        results, elapsed_time, weights_dict
    """
    emnist_dir = Path(emnist_dir)
    work_dir = Path(work_dir)
    
    # Disable integrations
    settings.update({'wandb': False, 'mlflow': False, 'comet': False})
    os.environ['WANDB_DISABLED'] = 'true'
    
    device_arg = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training on device: {device_arg}")
    
    # Load base model
    model = YOLO('yolov8n-cls.pt')
    
    import time
    start_t = time.time()
    results = model.train(
        data=str(emnist_dir),
        epochs=epochs,
        patience=3,  # Early stopping
        imgsz=28,  # EMNIST images are 28x28
        batch=batch,
        project=str(work_dir / 'runs'),
        name='emnist_benchmark',
        device=device_arg,
        plots=True,
        val=True
    )
    elapsed = time.time() - start_t
    
    print(f"✅ EMNIST training finished in {elapsed:.2f}s")
    
    # Find best.pt and last.pt
    weights = {}
    import glob
    glob_path = os.path.join(work_dir / 'runs', '**', 'emnist_benchmark', 'weights', '*.pt')
    found = glob.glob(glob_path, recursive=True)
    for f in found:
        if os.path.basename(f) in ('best.pt', 'last.pt'):
            weights[os.path.basename(f)] = f
    
    return results, elapsed, weights


def evaluate_cross_dataset(model_path: str, custom_dataset_dir: str, device: str = None):
    """Evaluate EMNIST-trained model on custom dataset (cross-dataset generalization).
    
    Args:
        model_path: Path to best.pt from EMNIST training
        custom_dataset_dir: Path to custom dataset (with train/val folders)
        device: Device to use ('cuda', 'cpu', or None for auto)
    
    Returns:
        Accuracy metrics dict
    """
    device = device or (0 if torch.cuda.is_available() else 'cpu')
    
    model = YOLO(model_path)
    print(f"Evaluating EMNIST-trained model on custom dataset...")
    
    metrics = model.val(data=custom_dataset_dir, split='val', device=device)
    
    print("\n====== 🌍 CROSS-DATASET EVALUATION ======")
    print(f"Training Source:   EMNIST")
    print(f"Testing Target:    Custom Dataset")
    print(f"Model:             {Path(model_path).name}")
    
    try:
        accuracy = metrics.top1
        print(f"Top-1 Accuracy:    {accuracy * 100:.2f}%")
        if accuracy < 0.85:
            print("✅ Accuracy drop suggests custom dataset has unique challenges (noise/blur/style).")
        else:
            print("✅ High accuracy indicates good dataset compatibility.")
    except Exception as e:
        print(f"Could not extract accuracy: {e}")
    
    print("=========================================")
    
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--emnist_dir', default='./workspace/emnist_yolo')
    parser.add_argument('--work_dir', default='./workspace')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--custom_dataset', help='Path to custom dataset for cross-eval (optional)')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation on existing best.pt')
    args = parser.parse_args()
    
    if not args.eval_only:
        download_and_prepare_emnist(args.emnist_dir)
        results, elapsed, weights = train_on_emnist(args.emnist_dir, args.work_dir, args.epochs, args.batch)
        
        if weights and 'best.pt' in weights:
            best_pt = weights['best.pt']
            if args.custom_dataset:
                evaluate_cross_dataset(best_pt, args.custom_dataset)
    else:
        # Load best.pt from default location and eval
        best_pt = Path(args.work_dir) / 'runs' / 'classify' / 'emnist_benchmark' / 'weights' / 'best.pt'
        if best_pt.exists() and args.custom_dataset:
            evaluate_cross_dataset(str(best_pt), args.custom_dataset)
        else:
            print(f"best.pt not found at {best_pt}")
