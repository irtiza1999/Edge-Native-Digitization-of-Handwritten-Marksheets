"""
Latency Comparison Benchmark: Full OCR Pipeline Analysis

This script benchmarks:
1. Individual model components (PaddleOCR, YOLO, MobileNet)
2. Full end-to-end pipelines (image → detection → recognition → Excel)

Outputs:
- Benchmark report JSON
- Comparison CSV
- Latency statistics and recommendations
"""

import json
import time
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def get_test_images(test_folder: str, max_images: int = 10) -> List[str]:
    """Get test images from workspace."""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    images = [
        str(p) for p in Path(test_folder).rglob('*')
        if p.suffix.lower() in valid_extensions
    ][:max_images]
    return sorted(images)


def benchmark_paddleocr(images: List[str], num_runs: int = 3) -> Dict:
    """Benchmark PaddleOCR pipeline using the existing run_paddle_ocr.py approach."""
    try:
        # Set protobuf fix for PaddleOCR
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        
        from paddleocr import PaddleOCR
        
        # Initialize PaddleOCR (will cache model after first load)
        print("   Initializing PaddleOCR (first run loads models ~1-2s)...")
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        
        latencies = []
        errors = 0
        
        for image_path in images:
            try:
                for _ in range(num_runs):
                    start = time.perf_counter()
                    result = ocr.ocr(image_path, cls=True)
                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)
            except Exception as e:
                errors += 1
        
        if latencies:
            return {
                'model': 'PaddleOCR',
                'images_processed': len(images) * num_runs - errors,
                'mean_latency_ms': np.mean(latencies) * 1000,
                'std_latency_ms': np.std(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'throughput_imgs_per_sec': 1000 / (np.mean(latencies) * 1000) if latencies else 0,
                'errors': errors,
                'latencies': latencies
            }
    except Exception as e:
        print(f"   Error benchmarking PaddleOCR: {e}")
        return None


def benchmark_tesseract(images: List[str], num_runs: int = 3) -> Dict:
    """Benchmark Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
        
        latencies = []
        errors = 0
        
        for image_path in images:
            try:
                img = Image.open(image_path)
                for _ in range(num_runs):
                    start = time.perf_counter()
                    text = pytesseract.image_to_string(img)
                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)
            except Exception as e:
                errors += 1
        
        if latencies:
            return {
                'model': 'Tesseract',
                'images_processed': len(images) * num_runs - errors,
                'mean_latency_ms': np.mean(latencies) * 1000,
                'std_latency_ms': np.std(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'throughput_imgs_per_sec': 1000 / (np.mean(latencies) * 1000) if latencies else 0,
                'errors': errors,
                'latencies': latencies
            }
    except ImportError:
        print("Tesseract not installed. Install with: pip install pytesseract")
        print("Also install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        return None
    except Exception as e:
        print(f"Error benchmarking Tesseract: {e}")
        return None


def benchmark_mobilenet(images: List[str], num_runs: int = 3) -> Dict:
    """Benchmark YOLOv8 MobileNet variant for detection (proxy for MobileNet performance)."""
    try:
        from ultralytics import YOLO
        
        # Load MobileNet-based YOLO model (lightweight variant)
        model = YOLO('yolov8n-cls.pt')  # nano classification model, closest to MobileNet efficiency
        
        latencies = []
        errors = 0
        
        for image_path in images:
            try:
                for _ in range(num_runs):
                    start = time.perf_counter()
                    results = model(image_path, verbose=False)
                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)
            except Exception as e:
                errors += 1
        
        if latencies:
            return {
                'model': 'MobileNet (YOLOv8n-cls)',
                'images_processed': len(images) * num_runs - errors,
                'mean_latency_ms': np.mean(latencies) * 1000,
                'std_latency_ms': np.std(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'throughput_imgs_per_sec': 1000 / (np.mean(latencies) * 1000) if latencies else 0,
                'errors': errors,
                'latencies': latencies
            }
    except Exception as e:
        print(f"Error benchmarking MobileNet: {e}")
        return None


def benchmark_yolo_detection(images: List[str], num_runs: int = 3) -> Dict:
    """Benchmark YOLOv8 detection pipeline (our project's YOLO setup)."""
    try:
        from ultralytics import YOLO
        
        # Load YOLOv8 nano detection model (typical for our use case)
        model = YOLO('yolov8n.pt')  # nano detection model
        
        latencies = []
        errors = 0
        
        for image_path in images:
            try:
                for _ in range(num_runs):
                    start = time.perf_counter()
                    results = model(image_path, verbose=False)
                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)
            except Exception as e:
                errors += 1
        
        if latencies:
            return {
                'model': 'YOLO (YOLOv8n-detect)',
                'images_processed': len(images) * num_runs - errors,
                'mean_latency_ms': np.mean(latencies) * 1000,
                'std_latency_ms': np.std(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'throughput_imgs_per_sec': 1000 / (np.mean(latencies) * 1000) if latencies else 0,
                'errors': errors,
                'latencies': latencies
            }
    except Exception as e:
        print(f"Error benchmarking YOLO: {e}")
        return None


def benchmark_full_paddleocr_pipeline(images: List[str], num_runs: int = 2) -> Dict:
    """Benchmark full end-to-end PaddleOCR pipeline: detection → recognition → Excel generation."""
    try:
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        from paddleocr import PaddleOCR
        import pandas as pd
        
        print("   Initializing PaddleOCR for full pipeline...")
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        
        latencies = []
        errors = 0
        excel_failures = 0
        last_error_message = None
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            for i, image_path in enumerate(images):
                try:
                    for run_idx in range(num_runs):
                        start = time.perf_counter()
                        
                        # 1. Detection + Recognition
                        result = ocr.ocr(image_path, cls=True)
                        
                        # 2. Group into table structure (flatten entries -> single text cell per row)
                        table_data = []
                        if result and result[0]:
                            for entry in result[0]:
                                try:
                                    text = entry[1][0] if entry[1] else ""
                                except Exception:
                                    text = ""
                                table_data.append([text])
                        
                        # 3. Generate Excel
                        if table_data:
                            try:
                                max_cols = max([len(r) for r in table_data])
                                table_data = [r + [''] * (max_cols - len(r)) for r in table_data]
                                df = pd.DataFrame(table_data)
                                excel_path = os.path.join(temp_dir, f"output_{i}_{run_idx}.xlsx")
                                df.to_excel(excel_path, index=False, header=False, engine='openpyxl')
                            except Exception as excel_err:
                                excel_failures += 1  # Track Excel write issues, but still measure latency
                        
                        elapsed = time.perf_counter() - start
                        latencies.append(elapsed)
                except Exception as e:
                    errors += 1
                    last_error_message = str(e)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        if latencies:
            return {
                'model': 'PaddleOCR Full Pipeline',
                'pipeline_stage': 'Detection → Recognition → Excel',
                'images_processed': len(images) * num_runs - errors,
                'mean_latency_ms': np.mean(latencies) * 1000,
                'std_latency_ms': np.std(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'throughput_imgs_per_sec': 1000 / (np.mean(latencies) * 1000) if latencies else 0,
                'errors': errors,
                'excel_failures': excel_failures,
                'latencies': latencies
            }
        # Return diagnostic even when no latencies were collected
        return {
            'model': 'PaddleOCR Full Pipeline',
            'pipeline_stage': 'Detection → Recognition → Excel',
            'images_processed': len(images) * num_runs - errors,
            'mean_latency_ms': None,
            'std_latency_ms': None,
            'p95_latency_ms': None,
            'p99_latency_ms': None,
            'min_latency_ms': None,
            'max_latency_ms': None,
            'throughput_imgs_per_sec': 0,
            'errors': errors,
            'excel_failures': excel_failures,
            'latencies': latencies,
            'error_message': last_error_message or 'No latencies collected (check OCR/Excel errors)'
        }
    except Exception as e:
        print(f"   Error benchmarking full pipeline: {e}")
        return None


def benchmark_yolo_plus_paddleocr_pipeline(images: List[str], num_runs: int = 2) -> Dict:
    """Benchmark YOLO detection + PaddleOCR recognition (hybrid approach)."""
    try:
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        from ultralytics import YOLO
        from paddleocr import PaddleOCR
        import pandas as pd
        
        print("   Initializing YOLO + PaddleOCR hybrid pipeline...")
        yolo_model = YOLO('yolov8n.pt')
        ocr = PaddleOCR(use_angle_cls=False, lang='ch')
        
        latencies = []
        errors = 0
        excel_failures = 0
        last_error_message = None
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            for i, image_path in enumerate(images):
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        errors += 1
                        continue
                    
                    for run_idx in range(num_runs):
                        start = time.perf_counter()
                        
                        # 1. YOLO Detection for table structure
                        yolo_results = yolo_model(image_path, verbose=False)
                        
                        # 2. PaddleOCR on full image for recognition
                        ocr_result = ocr.ocr(image_path, cls=False)
                        
                        # 3. Build table from OCR results
                        table_data = []
                        if ocr_result and ocr_result[0]:
                            for line in ocr_result[0]:
                                row_data = []
                                for word_info in line:
                                    text = word_info[1][0] if word_info[1] else ""
                                    row_data.append(text)
                                if row_data:
                                    table_data.append(row_data)
                        
                        # 4. Generate Excel
                        if table_data:
                            try:
                                max_cols = max([len(r) for r in table_data])
                                table_data = [r + [''] * (max_cols - len(r)) for r in table_data]
                                df = pd.DataFrame(table_data)
                                excel_path = os.path.join(temp_dir, f"output_hybrid_{i}_{run_idx}.xlsx")
                                df.to_excel(excel_path, index=False, header=False, engine='openpyxl')
                            except Exception as excel_err:
                                excel_failures += 1  # Track Excel write issues, but still measure latency
                        
                        elapsed = time.perf_counter() - start
                        latencies.append(elapsed)
                except Exception as e:
                    errors += 1
                    last_error_message = str(e)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        if latencies:
            return {
                'model': 'YOLO + PaddleOCR Hybrid',
                'pipeline_stage': 'YOLO Detection + OCR Recognition → Excel',
                'images_processed': len(images) * num_runs - errors,
                'mean_latency_ms': np.mean(latencies) * 1000,
                'std_latency_ms': np.std(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'throughput_imgs_per_sec': 1000 / (np.mean(latencies) * 1000) if latencies else 0,
                'errors': errors,
                'excel_failures': excel_failures,
                'latencies': latencies
            }
        # Return diagnostic even when no latencies were collected
        return {
            'model': 'YOLO + PaddleOCR Hybrid',
            'pipeline_stage': 'YOLO Detection + PaddleOCR → Excel',
            'images_processed': len(images) * num_runs - errors,
            'mean_latency_ms': None,
            'std_latency_ms': None,
            'p95_latency_ms': None,
            'p99_latency_ms': None,
            'min_latency_ms': None,
            'max_latency_ms': None,
            'throughput_imgs_per_sec': 0,
            'errors': errors,
            'excel_failures': excel_failures,
            'latencies': latencies,
            'error_message': last_error_message or 'No latencies collected (check OCR/Excel errors)'
        }
    except Exception as e:
        print(f"   Error benchmarking hybrid pipeline: {e}")
        return None


def benchmark_mobilenet_full_pipeline(images: List[str], num_runs: int = 2) -> Dict:
    """Benchmark MobileNet full pipeline: classification → Excel generation."""
    try:
        from ultralytics import YOLO
        import pandas as pd
        
        print("   Initializing MobileNet for full pipeline...")
        model = YOLO('yolov8n-cls.pt')
        
        latencies = []
        errors = 0
        excel_failures = 0
        last_error_message = None
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            for i, image_path in enumerate(images):
                try:
                    for run_idx in range(num_runs):
                        start = time.perf_counter()
                        
                        # 1. Classification
                        results = model(image_path, verbose=False)
                        
                        # 2. Extract predictions
                        table_data = []
                        if results and len(results) > 0:
                            result = results[0]
                            if hasattr(result, 'probs') and result.probs is not None:
                                top5_indices = result.probs.top5
                                top5_conf = result.probs.top5conf.tolist()
                                names = result.names
                                
                                for idx, conf in zip(top5_indices, top5_conf):
                                    class_name = names[int(idx)]
                                    table_data.append([class_name, f"{conf:.4f}"])
                        
                        # 3. Generate Excel
                        if table_data:
                            try:
                                df = pd.DataFrame(table_data, columns=['Class', 'Confidence'])
                                excel_path = os.path.join(temp_dir, f"mobilenet_{i}_{run_idx}.xlsx")
                                df.to_excel(excel_path, index=False, engine='openpyxl')
                            except Exception as excel_err:
                                excel_failures += 1
                        
                        elapsed = time.perf_counter() - start
                        latencies.append(elapsed)
                except Exception as e:
                    errors += 1
                    last_error_message = str(e)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        if latencies:
            return {
                'model': 'MobileNet Full Pipeline',
                'pipeline_stage': 'Classification → Excel',
                'images_processed': len(images) * num_runs - errors,
                'mean_latency_ms': np.mean(latencies) * 1000,
                'std_latency_ms': np.std(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'throughput_imgs_per_sec': 1000 / (np.mean(latencies) * 1000) if latencies else 0,
                'errors': errors,
                'excel_failures': excel_failures,
                'latencies': latencies
            }
        return {
            'model': 'MobileNet Full Pipeline',
            'pipeline_stage': 'Classification → Excel',
            'images_processed': len(images) * num_runs - errors,
            'mean_latency_ms': None,
            'std_latency_ms': None,
            'p95_latency_ms': None,
            'p99_latency_ms': None,
            'min_latency_ms': None,
            'max_latency_ms': None,
            'throughput_imgs_per_sec': 0,
            'errors': errors,
            'excel_failures': excel_failures,
            'latencies': latencies,
            'error_message': last_error_message or 'No latencies collected'
        }
    except Exception as e:
        print(f"   Error benchmarking MobileNet full pipeline: {e}")
        return None


def benchmark_tesseract_full_pipeline(images: List[str], num_runs: int = 2) -> Dict:
    """Benchmark Tesseract full pipeline: OCR → Excel generation."""
    try:
        import pytesseract
        from PIL import Image
        import pandas as pd
        
        print("   Initializing Tesseract for full pipeline...")
        
        latencies = []
        errors = 0
        excel_failures = 0
        last_error_message = None
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            for i, image_path in enumerate(images):
                try:
                    for run_idx in range(num_runs):
                        start = time.perf_counter()
                        
                        # 1. OCR with Tesseract
                        img = Image.open(image_path)
                        text = pytesseract.image_to_string(img)
                        
                        # 2. Parse text into table structure
                        lines = [line.strip() for line in text.split('\n') if line.strip()]
                        table_data = [[line] for line in lines]
                        
                        # 3. Generate Excel
                        if table_data:
                            try:
                                df = pd.DataFrame(table_data, columns=['Text'])
                                excel_path = os.path.join(temp_dir, f"tesseract_{i}_{run_idx}.xlsx")
                                df.to_excel(excel_path, index=False, engine='openpyxl')
                            except Exception as excel_err:
                                excel_failures += 1
                        
                        elapsed = time.perf_counter() - start
                        latencies.append(elapsed)
                except Exception as e:
                    errors += 1
                    last_error_message = str(e)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        if latencies:
            return {
                'model': 'Tesseract Full Pipeline',
                'pipeline_stage': 'OCR → Excel',
                'images_processed': len(images) * num_runs - errors,
                'mean_latency_ms': np.mean(latencies) * 1000,
                'std_latency_ms': np.std(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'throughput_imgs_per_sec': 1000 / (np.mean(latencies) * 1000) if latencies else 0,
                'errors': errors,
                'excel_failures': excel_failures,
                'latencies': latencies
            }
        return {
            'model': 'Tesseract Full Pipeline',
            'pipeline_stage': 'OCR → Excel',
            'images_processed': len(images) * num_runs - errors,
            'mean_latency_ms': None,
            'std_latency_ms': None,
            'p95_latency_ms': None,
            'p99_latency_ms': None,
            'min_latency_ms': None,
            'max_latency_ms': None,
            'throughput_imgs_per_sec': 0,
            'errors': errors,
            'excel_failures': excel_failures,
            'latencies': latencies,
            'error_message': last_error_message or 'No latencies collected'
        }
    except Exception as e:
        print(f"   Error benchmarking Tesseract full pipeline: {e}")
        return None


def save_results(results: List[Dict], output_dir: str):
    """Save benchmark results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for CSV (exclude raw latencies list)
    csv_data = []
    for result in results:
        if result:
            row = {k: v for k, v in result.items() if k != 'latencies'}
            csv_data.append(row)
    
    # Save JSON with full details
    json_path = os.path.join(output_dir, 'latency_comparison_report.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[OK] Saved JSON report: {json_path}")
    
    # Save CSV for easy comparison
    if csv_data:
        csv_path = os.path.join(output_dir, 'latency_comparison.csv')
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved CSV comparison: {csv_path}")
        
        # Print summary table
        print("\n" + "="*100)
        print("LATENCY COMPARISON SUMMARY")
        print("="*100)
        summary_df = df[[
            'model', 'images_processed', 'mean_latency_ms', 'std_latency_ms',
            'p95_latency_ms', 'p99_latency_ms', 'throughput_imgs_per_sec'
        ]].round(2)
        print(summary_df.to_string(index=False))
        print("="*100)
        
        return df
    return None


def main():
    """Run latency comparison benchmark."""
    print("Handwritten Table OCR - Latency Comparison Benchmark")
    print("="*100)
    
    # Get test images from workspace
    test_folder = Path(__file__).parent.parent / "workspace" / "ss thesis"
    if not test_folder.exists():
        print(f"Error: Test folder not found at {test_folder}")
        sys.exit(1)
    
    images = get_test_images(str(test_folder), max_images=10)
    if not images:
        print(f"Error: No images found in {test_folder}")
        sys.exit(1)
    
    print(f"\nFound {len(images)} test images")
    print(f"Running benchmarks (individual models: 3 runs, full pipelines: 2 runs)...\n")
    
    # Benchmark all models
    results = []
    
    # ========== INDIVIDUAL MODELS ==========
    print("="*100)
    print("INDIVIDUAL MODEL COMPONENTS")
    print("="*100)
    
    print("\n1. Benchmarking PaddleOCR (model only)...")
    paddle_result = benchmark_paddleocr(images, num_runs=3)
    if paddle_result:
        results.append(paddle_result)
        print(f"   [OK] Mean latency: {paddle_result['mean_latency_ms']:.2f} ms")
    else:
        print("   [FAIL] Failed to benchmark PaddleOCR")
    
    print("\n2. Benchmarking Tesseract...")
    tesseract_result = benchmark_tesseract(images, num_runs=3)
    if tesseract_result:
        results.append(tesseract_result)
        print(f"   [OK] Mean latency: {tesseract_result['mean_latency_ms']:.2f} ms")
    else:
        print("   [FAIL] Tesseract not available or failed")
    
    print("\n3. Benchmarking YOLO Detection (YOLOv8n-detect)...")
    yolo_result = benchmark_yolo_detection(images, num_runs=3)
    if yolo_result:
        results.append(yolo_result)
        print(f"   [OK] Mean latency: {yolo_result['mean_latency_ms']:.2f} ms")
    else:
        print("   [FAIL] Failed to benchmark YOLO")
    
    print("\n4. Benchmarking MobileNet (YOLOv8n-cls)...")
    mobilenet_result = benchmark_mobilenet(images, num_runs=3)
    if mobilenet_result:
        results.append(mobilenet_result)
        print(f"   [OK] Mean latency: {mobilenet_result['mean_latency_ms']:.2f} ms")
    else:
        print("   [FAIL] Failed to benchmark MobileNet")
    
    # ========== FULL PIPELINES ==========
    print("\n" + "="*100)
    print("FULL END-TO-END PIPELINES (Detection → Recognition → Excel)")
    print("="*100)
    
    print("\n5. Benchmarking PaddleOCR Full Pipeline (detection + recognition + Excel)...")
    full_paddle = benchmark_full_paddleocr_pipeline(images, num_runs=2)
    if full_paddle:
        results.append(full_paddle)
        if full_paddle.get('mean_latency_ms') is not None:
            print(f"   [OK] Mean latency: {full_paddle['mean_latency_ms']:.2f} ms")
        else:
            msg = full_paddle.get('error_message', 'Unknown issue')
            print(f"   [FAIL] PaddleOCR full pipeline incomplete: {msg}")
    else:
        print("   [FAIL] Failed to benchmark PaddleOCR full pipeline")
    
    print("\n6. Benchmarking YOLO + PaddleOCR Hybrid Pipeline...")
    hybrid_pipeline = benchmark_yolo_plus_paddleocr_pipeline(images, num_runs=2)
    if hybrid_pipeline:
        results.append(hybrid_pipeline)
        if hybrid_pipeline.get('mean_latency_ms') is not None:
            print(f"   [OK] Mean latency: {hybrid_pipeline['mean_latency_ms']:.2f} ms")
        else:
            msg = hybrid_pipeline.get('error_message', 'Unknown issue')
            print(f"   [FAIL] Hybrid pipeline incomplete: {msg}")
    else:
        print("   [FAIL] Failed to benchmark hybrid pipeline")
    
    print("\n7. Benchmarking MobileNet Full Pipeline (classification + Excel)...")
    mobilenet_full = benchmark_mobilenet_full_pipeline(images, num_runs=2)
    if mobilenet_full:
        results.append(mobilenet_full)
        if mobilenet_full.get('mean_latency_ms') is not None:
            print(f"   [OK] Mean latency: {mobilenet_full['mean_latency_ms']:.2f} ms")
        else:
            msg = mobilenet_full.get('error_message', 'Unknown issue')
            print(f"   [FAIL] MobileNet full pipeline incomplete: {msg}")
    else:
        print("   [FAIL] Failed to benchmark MobileNet full pipeline")
    
    print("\n8. Benchmarking Tesseract Full Pipeline (OCR + Excel)...")
    tesseract_full = benchmark_tesseract_full_pipeline(images, num_runs=2)
    if tesseract_full:
        results.append(tesseract_full)
        if tesseract_full.get('mean_latency_ms') is not None:
            print(f"   [OK] Mean latency: {tesseract_full['mean_latency_ms']:.2f} ms")
        else:
            msg = tesseract_full.get('error_message', 'Unknown issue')
            print(f"   [FAIL] Tesseract full pipeline incomplete: {msg}")
    else:
        print("   [FAIL] Failed to benchmark Tesseract full pipeline")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "outputs" / "latency_comparison"
    df = save_results(results, str(output_dir))
    
    # Print speedup comparisons
    if df is not None and len(df) > 1:
        print("\n" + "="*100)
        print("SPEEDUP COMPARISONS (relative to slowest model)")
        print("="*100)
        
        baseline_latency = df['mean_latency_ms'].max()
        baseline_model = df.loc[df['mean_latency_ms'].idxmax(), 'model']
        
        print(f"\nBaseline (slowest): {baseline_model} @ {baseline_latency:.2f} ms\n")
        
        for idx, row in df.iterrows():
            speedup = baseline_latency / row['mean_latency_ms']
            print(f"{row['model']:40s}: {row['mean_latency_ms']:8.2f} ms  |  Speedup: {speedup:6.2f}x")
        
        print("="*100)


if __name__ == '__main__':
    main()
