"""EMNIST benchmarking runner."""
import argparse
from src.emnist import download_and_prepare_emnist, train_on_emnist, evaluate_cross_dataset
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EMNIST benchmark: download, train, evaluate')
    parser.add_argument('--emnist_dir', default='./workspace/emnist_yolo', help='EMNIST dataset directory')
    parser.add_argument('--work_dir', default='./workspace', help='Workspace directory for runs')
    parser.add_argument('--epochs', type=int, default=25, help='Training epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--custom_dataset', help='Path to custom dataset for cross-eval')
    parser.add_argument('--eval_only', action='store_true', help='Only eval existing best.pt')
    args = parser.parse_args()
    
    if not args.eval_only:
        print("\n🧪 EMNIST Benchmark: Download & Train")
        print("=" * 50)
        
        # Download and prepare
        emnist_dir = download_and_prepare_emnist(args.emnist_dir)
        
        # Train
        results, elapsed, weights = train_on_emnist(
            emnist_dir, 
            args.work_dir, 
            args.epochs, 
            args.batch
        )
        
        # Evaluate on custom dataset if provided
        if weights and 'best.pt' in weights and args.custom_dataset:
            print("\n🌍 Cross-Dataset Evaluation")
            print("=" * 50)
            evaluate_cross_dataset(weights['best.pt'], args.custom_dataset)
    else:
        # Eval only
        best_pt = Path(args.work_dir) / 'runs' / 'classify' / 'emnist_benchmark' / 'weights' / 'best.pt'
        if best_pt.exists() and args.custom_dataset:
            print("\n🌍 Cross-Dataset Evaluation (Existing Model)")
            print("=" * 50)
            evaluate_cross_dataset(str(best_pt), args.custom_dataset)
        else:
            print(f"❌ Model not found at {best_pt}")
