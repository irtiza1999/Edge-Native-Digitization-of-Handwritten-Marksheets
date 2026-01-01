"""Training utilities: YAML creation and YOLO training wrapper with analytics."""
from pathlib import Path
import yaml
import time
import torch
from ultralytics import YOLO, settings
import os
from src.analytics import TrainingAnalytics


def create_yolo_yaml(num_classes: int, yaml_path: str, work_dir: str):
    custom = {
        'nc': num_classes,
        'backbone': [
            [-1, 1, 'Conv', [64, 3, 2]],
            [-1, 1, 'Conv', [128, 3, 2]],
            [-1, 3, 'C2f', [128, True]],
            [-1, 1, 'Conv', [256, 3, 2]],
            [-1, 6, 'C2f', [256, True]],
            [-1, 1, 'Conv', [512, 3, 2]],
            [-1, 6, 'C2f', [512, True]],
            [-1, 1, 'Conv', [1024, 3, 2]],
            [-1, 3, 'C2f', [1024, True]],
        ],
        'head': [
            [-1, 1, 'Classify', ['nc']]
        ]
    }
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(custom, f)
    return yaml_path


def train_yolo(yaml_path: str, data_root: str, epochs=20, imgsz=64, batch=16, device=None, name='run', analytics=True):
    settings.update({'wandb': False, 'mlflow': False, 'comet': False})
    os.environ['WANDB_DISABLED'] = 'true'

    device_arg = device if device is not None else (0 if torch.cuda.is_available() else 'cpu')
    
    # Initialize analytics if requested
    analyzer = TrainingAnalytics() if analytics else None

    # Ensure ultralytics save_dir points to work_dir/runs
    # The caller can set settings.save_dir externally; YAML path should be prepared already
    model = YOLO('yolov8n-cls.pt')
    
    print("🚀 Starting YOLO training with analytics...")
    start_t = time.time()
    epoch_start_t = start_t
    
    try:
        results = model.train(
            data=data_root,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device_arg,
            name=name,
            verbose=True
        )
        elapsed = time.time() - start_t
        
        # Log final metrics (if analyzer available)
        if analyzer:
            # Note: YOLO returns results object with results list
            # Each result has loss data; extract what we can
            if hasattr(results, 'results'):
                for i, result in enumerate(results.results):
                    if hasattr(result, 'loss'):
                        analyzer.log_epoch(
                            epoch=i,
                            train_loss=float(result.loss) if hasattr(result, 'loss') else None,
                            time_per_epoch=elapsed / len(results.results) if results.results else None
                        )
            analyzer.plot_training_curves()
            analyzer.plot_epoch_timing()
            analyzer.save_metrics_json()
            analyzer.generate_training_summary()
    
    except Exception as e:
        elapsed = time.time() - start_t
        print(f"⚠️  Training interrupted: {e}")
        results = None

    # After training, locate produced weights (best.pt, last.pt) and return their paths if present
    # Ultraytics saves runs under settings.save_dir (default) or project param; search recursively
    run_root = settings.save_dir if hasattr(settings, 'save_dir') and settings.save_dir else None
    weights = {}
    if run_root:
        import glob
        glob_path = os.path.join(run_root, '**', name, 'weights', '*.pt')
        found = glob.glob(glob_path, recursive=True)
        for f in found:
            if os.path.basename(f) in ('best.pt', 'last.pt'):
                weights[os.path.basename(f)] = f

    return results, elapsed, weights
