"""PaddleOCR table detection + TrOCR recognition pipeline.
Ported from notebook Cell 13-14.
Groups detected text boxes into rows and recognizes each cell with TrOCR.
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
import glob
from PIL import Image
from pathlib import Path
from tqdm import tqdm
# Note: PaddleOCR, transformers, and scipy/sklearn are optional and heavy.
# We lazy-import them inside the pipeline function to avoid long import chains.


def group_boxes_into_rows(dt_boxes, y_threshold: int = 20):
    """Group bounding boxes into rows based on Y-coordinate overlap.
    
    Args:
        dt_boxes: List of [box, confidence] tuples from PaddleOCR
        y_threshold: Vertical distance threshold for grouping boxes into same row
    
    Returns:
        List of rows, where each row is a list of [box, confidence] tuples sorted by X
    """
    if not dt_boxes:
        return []
    
    # Sort by top-left Y-coordinate
    dt_boxes_sorted = sorted(dt_boxes, key=lambda x: x[0][0][1])
    
    rows = []
    current_row = []
    last_y_center = (dt_boxes_sorted[0][0][0][1] + dt_boxes_sorted[0][0][2][1]) / 2
    
    for box_pack in dt_boxes_sorted:
        box = box_pack[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        # Calculate center Y of this box
        y_center = (box[0][1] + box[2][1]) / 2
        
        # If close to current row's Y, add to row
        if abs(y_center - last_y_center) < y_threshold:
            current_row.append(box_pack)
        else:
            # Finish old row, sort it by X, and start new one
            if current_row:
                current_row_sorted = sorted(current_row, key=lambda x: x[0][0][0])
                rows.append(current_row_sorted)
            
            current_row = [box_pack]
            last_y_center = y_center
    
    # Append last row
    if current_row:
        current_row_sorted = sorted(current_row, key=lambda x: x[0][0][0])
        rows.append(current_row_sorted)
    
    return rows


def recognize_crop_with_trocr(img, box, trocr_processor, trocr_model, device):
    """Crop image based on box and run TrOCR recognition.
    
    Args:
        img: OpenCV image (BGR)
        box: Bounding box [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        trocr_processor: TrOCRProcessor
        trocr_model: VisionEncoderDecoderModel
        device: torch device
    
    Returns:
        Recognized text string
    """
    # Extract box coordinates
    x_coords = [int(p[0]) for p in box]
    y_coords = [int(p[1]) for p in box]
    
    x_min, x_max = max(0, min(x_coords)), max(x_coords)
    y_min, y_max = max(0, min(y_coords)), max(y_coords)
    
    # Add padding
    pad = 5
    h, w = img.shape[:2]
    y_min = max(0, y_min - pad)
    y_max = min(h, y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(w, x_max + pad)
    
    crop = img[y_min:y_max, x_min:x_max]
    
    if crop.size == 0:
        return ""
    
    # Convert BGR to RGB for PIL
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb).convert("RGB")
    
    # TrOCR inference
    pixel_values = trocr_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
    text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return text


def process_folder_with_paddle_trocr(input_folder: str, output_folder: str, 
                                      use_trocr: bool = True, 
                                      device: str = None):
    """Process folder of images: detect text boxes with PaddleOCR, recognize with TrOCR.
    
    Args:
        input_folder: Path to folder containing images
        output_folder: Path to save Excel files with extracted text
        use_trocr: If True, use TrOCR for recognition; if False, use PaddleOCR recognition
        device: torch device ('cuda' or 'cpu'); auto-detected if None
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load PaddleOCR (detection only) with graceful handling when not installed
    print("Loading PaddleOCR for text detection...")
    # Workaround for incompatible protobuf versions when pre-generated _pb2 files are present
    # Setting PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python forces the pure-Python implementation
    # (slower, but avoids the 'Descriptors cannot be created directly' error).
    if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() != "python":
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        print("Warning: set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python to avoid protobuf descriptor errors.")

    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as e:
        # Provide actionable install instructions depending on CPU/GPU
        msg = (
            "PaddleOCR (and/or PaddlePaddle) is not installed.\n"
            "Install instructions:\n"
            " - CPU-only Windows: `pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html`\n"
            " - GPU (CUDA) Windows: follow instructions at https://www.paddlepaddle.org.cn/install/quick\n"
            " - Then install PaddleOCR: `pip install paddleocr`\n\n"
            "If you prefer not to install PaddlePaddle, run with `--no_trocr` and use other OCR options or set the env var `DISABLE_MODEL_SOURCE_CHECK=True` to bypass online model checks.\n"
            "Error details: " + str(e)
        )
        print(msg)
        return

    # Initialize PaddleOCR with retries to handle transient download/network errors
    det_model = None
    import time as _time
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            det_model = PaddleOCR(use_textline_orientation=False, lang='en')
            break
        except Exception as e:
            print(f"PaddleOCR model init/download failed (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                print("Retrying in 5 seconds...")
                _time.sleep(5)
            else:
                print("Failed to initialize PaddleOCR after multiple attempts.\nPlease check your network connection or pre-download the PaddleOCR models to your cache (e.g. C:\\Users\\<you>\\.paddleocr).\nYou can also set the environment variable DISABLE_MODEL_SOURCE_CHECK=True to bypass automatic online checks (use with caution).")
                return
    
    # Load TrOCR if requested (guard against missing deps like SentencePiece)
    trocr_processor = None
    trocr_model = None
    if use_trocr:
        print("Loading TrOCR for text recognition...")
        # Check sentencepiece dependency first
        try:
            import sentencepiece  # type: ignore
        except ModuleNotFoundError:
            print("SentencePiece is not installed. TrOCR tokenizers require it.")
            print("Install it with: pip install sentencepiece")
            print("Falling back to PaddleOCR recognition. To force TrOCR, install dependencies and re-run.")
            use_trocr = False

    if use_trocr:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=False)
            trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten').to(device)
        except Exception as e:
            # Provide actionable guidance instead of crashing
            print("Error loading TrOCR model/tokenizer:", e)
            print("Common fixes: ensure `sentencepiece` and `protobuf` are installed, and you have internet access to download the model.")
            print("You can install the dependencies with: pip install sentencepiece protobuf transformers")
            print("Falling back to PaddleOCR recognition for this run.")
            use_trocr = False
    
    # Find images
    image_files = glob.glob(str(input_folder / '*.*'))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images. Starting processing...")
    
    # Helper to handle different PaddleOCR API versions
    def paddle_predict(model, img_path, det=True, rec=False):
        try:
            if hasattr(model, 'predict'):
                return model.predict(img_path)
            if hasattr(model, 'ocr'):
                # paddleocr.ocr returns list of tuples depending on rec/det flags
                return model.ocr(img_path, det=det, rec=rec)
            # fallback: try calling the object
            return model(img_path)
        except Exception:
            return []

    for img_path in tqdm(image_files):
        fname = os.path.basename(img_path)
        
        try:
            # 1. Detect text boxes with PaddleOCR using compatible API
            result = paddle_predict(det_model, img_path, det=True, rec=True)
            
            if not result or not result[0]:
                print(f"No text detected in {fname}")
                continue
            
            # Parse boxes (result[0] is list of [box, text, confidence] or similar)
            raw_boxes = []
            for line in result[0]:
                # Each line is typically [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], text, confidence
                if isinstance(line, (list, tuple)) and len(line) >= 1:
                    box = line[0] if isinstance(line[0], list) else line
                    raw_boxes.append([box, 0])
            
            # 2. Group into rows
            rows = group_boxes_into_rows(raw_boxes, y_threshold=20)
            
            # 3. Recognize each box
            img_cv = cv2.imread(img_path)
            table_data = []
            
            for row_boxes in rows:
                row_text = []
                for box_pack in row_boxes:
                    box = box_pack[0]
                    
                    if use_trocr:
                        text = recognize_crop_with_trocr(img_cv, box, trocr_processor, trocr_model, device)
                    else:
                        # Fallback to PaddleOCR recognition — use full prediction on cropped image
                        try:
                            x_coords = [int(p[0]) for p in box]
                            y_coords = [int(p[1]) for p in box]
                            x_min, x_max = max(0, min(x_coords)), max(x_coords)
                            y_min, y_max = max(0, min(y_coords)), max(y_coords)
                            pad = 5
                            h, w = img_cv.shape[:2]
                            y_min = max(0, y_min - pad)
                            y_max = min(h, y_max + pad)
                            x_min = max(0, x_min - pad)
                            x_max = min(w, x_max + pad)
                            crop = img_cv[y_min:y_max, x_min:x_max]
                            if crop.size > 0:
                                # Use a portable temp directory (cross-platform)
                                import tempfile
                                with tempfile.TemporaryDirectory() as tmpdir:
                                    crop_path = os.path.join(tmpdir, f"crop_{len(row_text)}.png")
                                    cv2.imwrite(crop_path, crop)
                                    result_rec = paddle_predict(det_model, crop_path, det=False, rec=True)
                                # result_rec formats vary by PaddleOCR version:
                                # - [[('text', score), ...]]  (rec only)
                                # - [[ [box], ('text', score) ], ...] (det+rec)
                                text = ""
                                try:
                                    if result_rec and len(result_rec) > 0:
                                        first = result_rec[0]
                                        # If first is a list of tuples like [('text', score), ...]
                                        if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (tuple, list)) and isinstance(first[0][0], str):
                                            # e.g. [[('I', 0.49)]] -> take tuple[0]
                                            text = first[0][0]
                                        else:
                                            # Try deeper inspection: some APIs return [[ [box, ('text',score)], ... ]]
                                            # attempt to find a (text,score) inside
                                            found = None
                                            for item in first:
                                                if isinstance(item, (list, tuple)):
                                                    # item could be [box, ('text',score)] or ('text',score)
                                                    if len(item) >= 2 and isinstance(item[-1], (tuple, list)) and isinstance(item[-1][0], str):
                                                        found = item[-1][0]
                                                        break
                                                    if isinstance(item[0], str):
                                                        found = item[0]
                                                        break
                                            if found:
                                                text = found
                                except Exception:
                                    text = ""
                            else:
                                text = ""
                        except Exception as e:
                            text = f"[Error: {str(e)[:30]}]"
                    
                    row_text.append(text)
                
                table_data.append(row_text)
            
            # 4. Save to Excel
            if table_data:
                max_cols = max([len(r) for r in table_data])
                table_data = [r + [''] * (max_cols - len(r)) for r in table_data]
                
                df = pd.DataFrame(table_data)
                out_path = output_folder / f"{os.path.splitext(fname)[0]}.xlsx"
                df.to_excel(out_path, index=False, header=False)
                # print(f"✅ Saved {out_path}")
        
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    
    try:
        resolved = output_folder.resolve()
    except Exception:
        resolved = output_folder
    print(f"\nDone! Results saved in {resolved}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True, help='Path to folder with images')
    parser.add_argument('--output_folder', required=True, help='Path to save Excel files')
    parser.add_argument('--no_trocr', action='store_true', help='Use PaddleOCR recognition instead of TrOCR')
    parser.add_argument('--device', default=None, help='Device: cuda or cpu')
    args = parser.parse_args()
    
    process_folder_with_paddle_trocr(
        args.input_folder, 
        args.output_folder,
        use_trocr=not args.no_trocr,
        device=args.device
    )
