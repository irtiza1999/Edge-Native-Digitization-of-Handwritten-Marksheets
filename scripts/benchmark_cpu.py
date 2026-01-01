"""CPU latency benchmark for inference with detailed analysis (simulates edge device).
Ported from notebook Cell 10, enhanced with visualization and metrics.
"""
import time
import torch
import glob
import os
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
import argparse
import numpy as np
from src.analytics import BenchmarkAnalytics


def benchmark_cpu_latency(model_path: str, test_image_path: str = None, iterations: int = 50):
    """Measure CPU inference latency on a test image with analytics.
    
    Args:
        model_path: Path to best.pt or similar model
        test_image_path: Path to test image. If None, searches workspace for any image.
        iterations: Number of inference iterations for averaging
    
    Returns:
        Dict with latency (ms), FPS, and metadata
    """
    print("🧪 ---- CPU INFERENCE LATENCY BENCHMARK ----")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Device: CPU (edge device simulation)")
    
    # Initialize analytics
    analyzer = BenchmarkAnalytics(output_dir="outputs/benchmark_analysis")
    
    # Resolve model path (fallback search if not found)
    if not os.path.exists(model_path):
        print(f"Warning: model not found at provided path: {model_path}")
        # search common locations for best.pt
        candidates = []
        # workspace checkpoints
        candidates += [str(p) for p in Path('workspace').glob('**/checkpoints/**/best.pt')]
        # workspace runs weights
        candidates += [str(p) for p in Path('workspace').glob('**/runs/**/weights/**/best.pt')]
        # runs folder at repo root
        candidates += [str(p) for p in Path('runs').glob('**/weights/**/best.pt')]
        # any best.pt in repo
        candidates += [str(p) for p in Path('.').glob('**/best.pt')]
        # dedupe while preserving order
        seen = set()
        candidates = [x for x in candidates if not (x in seen or seen.add(x))]
        if candidates:
            model_path = candidates[0]
            print(f"Auto-selected model: {model_path}")
        else:
            print("No candidate checkpoints found. Searched common locations under 'workspace' and 'runs'.")
            print("Please pass a valid --model path. Example:")
            print("  python scripts/benchmark_cpu.py --model .\\workspace\\runs\\emnist_benchmark2\\weights\\best.pt")
            return {}

    # Load model
    model = YOLO(model_path)
    
    # Find test image if not provided
    if not test_image_path:
        print("Searching for test image in workspace...")
        search_paths = [
            './workspace/dataset_augmented/val/**/*.jpg',
            './workspace/emnist_yolo/val/**/*.jpg',
            './**/*.jpg',
        ]
        for pattern in search_paths:
            candidates = glob.glob(pattern, recursive=True)
            if candidates:
                test_image_path = candidates[0]
                break
    
    if not test_image_path or not os.path.exists(test_image_path):
        print(f"❌ No test image found. Please provide --test_image_path")
        return {}
    
    print(f"📸 Test image: {test_image_path}")
    
    # Warmup (discarded)
    print("🔥 Warming up CPU...")
    for _ in range(3):
        model.predict(test_image_path, device='cpu', verbose=False)
    
    # Benchmark with latency tracking
    print(f"⏱️  Running {iterations} iterations...")
    latencies = []
    start_t = time.time()
    for i in range(iterations):
        iter_start = time.time()
        model.predict(test_image_path, device='cpu', verbose=False)
        iter_time = (time.time() - iter_start) * 1000  # Convert to ms
        latencies.append(iter_time)
    end_t = time.time()
    
    total_time = end_t - start_t
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    fps = 1000 / avg_latency
    
    # Add to analyzer
    analyzer.add_timing(
        model_name=os.path.basename(model_path),
        latencies=latencies,
        throughput=fps,
        batch_size=1
    )
    
    print("\n====== ⚡ CPU LATENCY REPORT ======")
    print(f"Device:                CPU")
    print(f"Model:                 {os.path.basename(model_path)}")
    print(f"Test Image:            {os.path.basename(test_image_path)}")
    print(f"Iterations:            {iterations}")
    print(f"Total Time:            {total_time:.2f}s")
    print(f"\n📊 Latency Statistics (ms):")
    print(f"  Mean:                {avg_latency:.3f}")
    print(f"  Std Dev:             {std_latency:.3f}")
    print(f"  Min:                 {min_latency:.3f}")
    print(f"  Max:                 {max_latency:.3f}")
    print(f"  P95:                 {p95_latency:.3f}")
    print(f"  P99:                 {p99_latency:.3f}")
    print(f"\n⚡ Throughput:          {fps:.2f} FPS")
    print("===================================\n")
    
    # Generate visualizations
    print("📈 Generating benchmark analysis...")
    analyzer.plot_latency_distribution()
    analyzer.plot_model_comparison()
    analyzer.generate_benchmark_report()
    
    results = {
        'model': model_path,
        'device': 'cpu',
        'mean_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'p95_latency_ms': p95_latency,
        'p99_latency_ms': p99_latency,
        'fps': fps,
        'iterations': iterations,
        'total_time': total_time,
        'latencies': latencies  # Full list for advanced analysis
    }
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .pt model file')
    parser.add_argument('--test_image', help='Path to test image (optional; searched if omitted)')
    parser.add_argument('--iterations', type=int, default=50, help='Number of inference iterations')
    args = parser.parse_args()
    
    results = benchmark_cpu_latency(args.model, args.test_image, args.iterations)
