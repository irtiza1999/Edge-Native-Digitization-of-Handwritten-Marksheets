"""Extract all training runs' args.yaml into a single CSV for easy comparison."""
import os
import yaml
import pandas as pd
from pathlib import Path

def extract_runs_csv(runs_dir='workspace/runs', output_csv='outputs/metrics_summary/report/training_runs_summary.csv'):
    """
    Extract all args.yaml from runs and create a comparison CSV.
    """
    runs_dir = Path(runs_dir)
    output_csv = Path(output_csv)
    
    # Ensure output dir exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    runs_data = []
    
    # Find all args.yaml files
    for args_file in sorted(runs_dir.glob('*/args.yaml')):
        run_name = args_file.parent.name
        
        try:
            with open(args_file, 'r') as f:
                args = yaml.safe_load(f)
            
            # Extract key hyperparameters for comparison
            extracted = {
                'run_name': run_name,
                'epochs': args.get('epochs', 'N/A'),
                'batch': args.get('batch', 'N/A'),
                'imgsz': args.get('imgsz', 'N/A'),
                'lr0': args.get('lr0', 'N/A'),
                'lrf': args.get('lrf', 'N/A'),
                'momentum': args.get('momentum', 'N/A'),
                'weight_decay': args.get('weight_decay', 'N/A'),
                'warmup_epochs': args.get('warmup_epochs', 'N/A'),
                'warmup_momentum': args.get('warmup_momentum', 'N/A'),
                'optimizer': args.get('optimizer', 'N/A'),
                'patience': args.get('patience', 'N/A'),
                'device': args.get('device', 'N/A'),
                'model': args.get('model', 'N/A'),
                'data': args.get('data', 'N/A'),
                'task': args.get('task', 'N/A'),
            }
            runs_data.append(extracted)
        except Exception as e:
            print(f"Error reading {args_file}: {e}")
    
    if runs_data:
        df = pd.DataFrame(runs_data)
        df.to_csv(output_csv, index=False)
        print(f"Extracted {len(runs_data)} runs to {output_csv}")
        print("\nPreview:")
        print(df.to_string(index=False))
        return output_csv
    else:
        print("No args.yaml files found.")
        return None

if __name__ == '__main__':
    extract_runs_csv()
