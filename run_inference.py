"""Run TrOCR inference over a folder of images and save excel outputs with detailed analysis.
Generates visualizations and metrics for inference results.
"""
import os
# Set protobuf implementation before importing any protobuf-dependent libraries
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
from src.trocr_runner import TrOCRRunner
from src.analytics import InferenceAnalytics
# Compatibility helper for PaddleOCR API differences (local copy)
def paddle_predict(model, img_path, det=True, rec=False):
    try:
        if hasattr(model, 'predict'):
            return model.predict(img_path)
        if hasattr(model, 'ocr'):
            return model.ocr(img_path, det=det, rec=rec)
        return model(img_path)
    except Exception:
        return []
from PIL import Image
import pandas as pd
import glob
import sys
import time
from paddleocr import PaddleOCR


def _ensure_dependencies():
    # Protobuf check made optional due to conda/venv PATH issues on Windows
    # TrOCR will attempt to load and provide better error if truly missing
    pass


def main():
    parser = argparse.ArgumentParser(description='TrOCR Inference with Analysis')
    parser.add_argument('--input_folder', required=True, help='Input folder with images')
    parser.add_argument('--output_folder', required=True, help='Output folder for results')
    parser.add_argument('--ground_truth_folder', help='Optional folder with ground truth text files')
    parser.add_argument('--save_visualizations', action='store_true', default=True,
                       help='Generate analysis visualizations')
    parser.add_argument('--no_trocr', action='store_true', default=False,
                       help='Use PaddleOCR instead of TrOCR (for torch <2.6 compatibility)')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    _ensure_dependencies()
    
    # Initialize analytics
    analyzer = InferenceAnalytics(output_dir="outputs/inference_analysis")
    
    # Use PaddleOCR or TrOCR based on flag
    if args.no_trocr:
        print("Using PaddleOCR for text recognition...")
        paddle_model = PaddleOCR(use_angle_cls=False, lang='en')
        runner = None
    else:
        # Use use_fast=False to avoid tokenizer fast-conversion issues on some environments
        runner = TrOCRRunner(use_fast=False)
        paddle_model = None

    images = glob.glob(os.path.join(args.input_folder, '*.*'))
    images = [p for p in images if p.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

    if not images:
        print(f"ERROR: No images found in {args.input_folder}")
        return

    print(f"Running inference on {len(images)} images...")
    
    start_time = time.time()
    results_data = []
    
    for idx, img_path in enumerate(images, 1):
        try:
            img_name = os.path.basename(img_path)
            img = Image.open(img_path).convert('RGB')
            
            t0 = time.time()
            
            # Recognize text using selected model
            if args.no_trocr:
                # PaddleOCR returns list of [[text, confidence], ...] for each detected region
                # Use PaddleOCR via compatibility helper
                ocr_result = paddle_predict(paddle_model, img_path, det=True, rec=True)
                # paddle_predict may return nested structures depending on PaddleOCR version.
                # Common shapes:
                #  - [[ [box, text, score], ... ]]  -> top-level list with one element (list of detections)
                #  - [ [text, score], ... ]         -> flat list of (text, score)
                # We'll handle both robustly.
                text = ""
                if ocr_result:
                    dets = ocr_result[0] if isinstance(ocr_result[0], (list, tuple)) and len(ocr_result) == 1 else ocr_result
                    texts = []
                    for det in dets:
                        # det may be [box, text, score] or [text, score] or an object
                        try:
                            if isinstance(det, (list, tuple)):
                                # prefer the second element when available
                                if len(det) >= 2 and isinstance(det[1], str):
                                    texts.append(det[1])
                                elif len(det) >= 3 and isinstance(det[2], str):
                                    texts.append(det[2])
                                else:
                                    # fallback: stringify
                                    texts.append(str(det))
                            else:
                                # object-like result (e.g. OCRResult) — stringify
                                texts.append(str(det))
                        except Exception:
                            continue
                    text = '\n'.join(texts)
                model_name = 'PaddleOCR'
            else:
                text = runner.recognize_image(img)
                model_name = 'TrOCR'
            
            inference_time = time.time() - t0
            
            # Try to load ground truth if provided
            gt_text = ""
            if args.ground_truth_folder:
                gt_path = os.path.join(args.ground_truth_folder, os.path.splitext(img_name)[0] + '.txt')
                if os.path.exists(gt_path):
                    with open(gt_path, 'r') as f:
                        gt_text = f.read().strip()
            
            # Save result to excel
            df = pd.DataFrame([[text]])
            out_xlsx = os.path.join(args.output_folder, os.path.splitext(img_name)[0] + '.xlsx')
            df.to_excel(out_xlsx, index=False, header=False)
            
            # Log to analyzer
            analyzer.add_prediction(
                pred_text=text,
                gt_text=gt_text,
                confidence=None,  # Models don't provide overall confidence
                image_path=img_path,
                model_name=model_name
            )
            
            results_data.append({
                'image': img_name,
                'prediction': text,
                'ground_truth': gt_text,
                'inference_time_ms': inference_time * 1000,
            })
            
            print(f"  [{idx:3d}/{len(images)}] OK  {img_name:30s} | Time: {inference_time*1000:6.1f}ms")
        
        except Exception as e:
            print(f"  [{idx:3d}/{len(images)}] ERR {img_name:30s} | Error: {str(e)[:40]}")
            continue
    
    elapsed = time.time() - start_time
    
    # Generate analysis
    print(f"\nGenerating analysis and visualizations...")
    
    # Save results CSV
    results_df = pd.DataFrame(results_data)
    results_csv = os.path.join(args.output_folder, "inference_results_summary.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")
    
    # Generate reports and visualizations
    analyzer.generate_inference_report()
    analyzer.save_detailed_results_csv()
    
    # Print summary
    print(f"\nInference Complete:")
    print(f"  Total Images: {len(images)}")
    print(f"  Total Time: {elapsed:.2f}s")
    print(f"  Avg Time/Image: {elapsed/len(images)*1000:.1f}ms")
    print(f"  Output folder: {os.path.abspath(args.output_folder)}")
    print(f"  Analysis folder: outputs/inference_analysis/")


if __name__ == '__main__':
    main()

