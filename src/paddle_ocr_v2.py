"""Improved handwritten table detection and OCR using PaddleOCR + TrOCR.
Handles table images by pre-processing, detecting cells via contours, and extracting text.
"""
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import torch
import glob
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def preprocess_image(img_path: str, enhance_contrast: bool = True):
    """Load and pre-process image for better table detection.
    
    Args:
        img_path: Path to image
        enhance_contrast: If True, apply CLAHE and morphological operations
    
    Returns:
        Preprocessed image (BGR)
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if enhance_contrast:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    
    # Apply slight Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Threshold to binary
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return binary, gray, img


def detect_table_cells(binary_img, min_cell_area: int = 100):
    """Detect table cells using contour detection.
    
    Args:
        binary_img: Binary image from preprocessing
        min_cell_area: Minimum cell area to filter noise
    
    Returns:
        List of bounding boxes (x, y, w, h) for detected cells
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Filter by size (avoid very small noise or very large regions)
        if min_cell_area < area < (binary_img.shape[0] * binary_img.shape[1] * 0.5):
            cells.append((x, y, w, h))
    
    # Sort cells left-to-right, top-to-bottom
    cells = sorted(cells, key=lambda c: (c[1], c[0]))
    return cells


def group_cells_into_rows(cells, y_threshold: int = 20):
    """Group detected cells into rows based on Y-coordinate.
    
    Args:
        cells: List of (x, y, w, h) bounding boxes
        y_threshold: Vertical distance threshold
    
    Returns:
        List of rows, each row is a list of cells sorted by X
    """
    if not cells:
        return []
    
    rows = []
    current_row = [cells[0]]
    current_y = cells[0][1]
    
    for cell in cells[1:]:
        if abs(cell[1] - current_y) < y_threshold:
            current_row.append(cell)
        else:
            # Sort row by X coordinate
            current_row.sort(key=lambda c: c[0])
            rows.append(current_row)
            current_row = [cell]
            current_y = cell[1]
    
    # Append last row
    if current_row:
        current_row.sort(key=lambda c: c[0])
        rows.append(current_row)
    
    return rows


def extract_text_from_cell(img_cv, cell_bbox, ocr_model, use_trocr: bool = False, 
                           trocr_processor=None, trocr_model=None, device=None):
    """Extract text from a single cell using OCR.
    
    Args:
        img_cv: Original color image (BGR)
        cell_bbox: (x, y, w, h) bounding box
        ocr_model: PaddleOCR model
        use_trocr: If True, use TrOCR; otherwise use PaddleOCR
        trocr_processor, trocr_model, device: TrOCR components
    
    Returns:
        Extracted text string
    """
    x, y, w, h = cell_bbox
    
    # Add small padding
    pad = 3
    h_img, w_img = img_cv.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    
    crop = img_cv[y1:y2, x1:x2]
    
    if crop.size == 0:
        return ""
    
    try:
        if use_trocr and trocr_processor and trocr_model:
            # TrOCR recognition
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb).convert("RGB")
            pixel_values = trocr_processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = trocr_model.generate(pixel_values)
            text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            # PaddleOCR recognition on cropped image
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            
            # Save temp and predict (use temp directory)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name
            cv2.imwrite(temp_path, crop)
            
            try:
                result = ocr_model.predict(temp_path)
                
                # Extract text from result
                if result and result[0]:
                    # result[0] contains [box, text, confidence]
                    text_list = [line[1] for line in result[0] if len(line) > 1]
                    text = " ".join(text_list)
                else:
                    text = ""
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return text.strip()
    except Exception as e:
        return f"[Error: {str(e)[:20]}]"


def process_handwritten_table(input_folder: str, output_folder: str, 
                               use_trocr: bool = True, device: str = None):
    """Process handwritten table images and extract data to Excel.
    
    Args:
        input_folder: Path to folder with table images
        output_folder: Path to save Excel files
        use_trocr: If True, use TrOCR; otherwise use PaddleOCR recognition
        device: torch device ('cuda' or 'cpu'); auto-detected if None
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load PaddleOCR
    print("🚀 Loading PaddleOCR for text recognition...")
    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError:
        print("PaddleOCR is not installed.")
        print("Install it with: pip install paddleocr paddlepaddle")
        return
    
    ocr_model = PaddleOCR(use_textline_orientation=False, lang='en')
    
    # Load TrOCR if requested
    trocr_processor = None
    trocr_model = None
    if use_trocr:
        print("🚀 Loading TrOCR for handwritten text recognition...")
        try:
            import sentencepiece
        except ModuleNotFoundError:
            print("SentencePiece not installed. Falling back to PaddleOCR.")
            use_trocr = False
        
        if use_trocr:
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', use_fast=False)
                trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten').to(device)
            except Exception as e:
                print(f"Error loading TrOCR: {e}. Falling back to PaddleOCR.")
                use_trocr = False
    
    # Find images
    image_files = glob.glob(str(input_folder / '*.*'))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"❌ No images found in {input_folder}")
        return
    
    print(f"✅ Found {len(image_files)} images. Processing tables...")
    
    for img_path in tqdm(image_files, desc="Processing"):
        fname = os.path.basename(img_path)
        
        try:
            # 1. Preprocess image
            binary, gray, img_color = preprocess_image(img_path, enhance_contrast=True)
            if binary is None:
                print(f"⚠️  Could not load {fname}")
                continue
            
            # 2. Detect table cells
            cells = detect_table_cells(binary, min_cell_area=50)
            if not cells:
                print(f"⚠️  No table cells detected in {fname}")
                continue
            
            # 3. Group cells into rows
            rows = group_cells_into_rows(cells, y_threshold=15)
            
            # 4. Extract text from each cell
            table_data = []
            for row_cells in rows:
                row_text = []
                for cell in row_cells:
                    text = extract_text_from_cell(
                        img_color, cell, ocr_model,
                        use_trocr=use_trocr,
                        trocr_processor=trocr_processor,
                        trocr_model=trocr_model,
                        device=device
                    )
                    row_text.append(text)
                table_data.append(row_text)
            
            # 5. Save to Excel
            if table_data:
                # Pad rows to same length
                max_cols = max([len(r) for r in table_data])
                table_data = [r + [''] * (max_cols - len(r)) for r in table_data]
                
                df = pd.DataFrame(table_data)
                out_path = output_folder / f"{os.path.splitext(fname)[0]}_extracted.xlsx"
                df.to_excel(out_path, index=False, header=False)
                print(f"✅ Saved {out_path.name}")
        
        except Exception as e:
            print(f"❌ Error processing {fname}: {str(e)[:50]}")
    
    print(f"\n🎉 Done! Results saved in {output_folder}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Handwritten Table OCR: detect table cells and extract text')
    parser.add_argument('--input_folder', required=True, help='Input folder with table images')
    parser.add_argument('--output_folder', required=True, help='Output folder for Excel files')
    parser.add_argument('--no_trocr', action='store_true', help='Use PaddleOCR recognition instead of TrOCR')
    parser.add_argument('--device', help='cuda or cpu (auto-detect if omitted)')
    args = parser.parse_args()
    
    print("\n📊 Handwritten Table Detection & OCR Pipeline")
    print("=" * 50)
    process_handwritten_table(
        args.input_folder,
        args.output_folder,
        use_trocr=not args.no_trocr,
        device=args.device
    )
