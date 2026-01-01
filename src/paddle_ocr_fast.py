"""Fast and reliable handwritten table detection and OCR using PaddleOCR.
Detects all text, groups into rows/columns, and exports to Excel.
"""
import os
import cv2
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm


def process_table_images(input_folder: str, output_folder: str):
    """Process handwritten table images and extract data to Excel using modern PaddleOCR API."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load PaddleOCR
    print("🚀 Loading PaddleOCR...")
    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError:
        print("PaddleOCR is not installed.")
        print("Install it with: pip install paddleocr paddlepaddle")
        return
    
    ocr_model = PaddleOCR(use_textline_orientation=False, lang='en')
    
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
            # Run PaddleOCR on image
            result = ocr_model.predict(img_path)
            
            if not result or len(result) == 0 or 'rec_texts' not in result[0]:
                print(f"⚠️  No text detected in {fname}")
                continue
            
            # Extract text and boxes from OCRResult
            ocr_result = result[0]
            rec_texts = ocr_result.get('rec_texts', [])
            rec_polys = ocr_result.get('rec_polys', [])
            
            if not rec_texts:
                print(f"⚠️  No text recognized in {fname}")
                continue
            
            # Build regions with bounding box info
            regions = []
            for i, (text, poly) in enumerate(zip(rec_texts, rec_polys)):
                if text and text.strip():
                    # Get bounding box from polygon
                    x_coords = poly[:, 0]
                    y_coords = poly[:, 1]
                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                    
                    regions.append({
                        'text': text.strip(),
                        'x': x_min,
                        'y': y_min,
                        'y_center': (y_min + y_max) / 2,
                        'x_center': (x_min + x_max) / 2,
                        'width': x_max - x_min,
                        'height': y_max - y_min
                    })
            
            if not regions:
                print(f"⚠️  No valid text regions in {fname}")
                continue
            
            # Sort by Y position (top to bottom), then X (left to right)
            regions.sort(key=lambda r: (r['y_center'], r['x_center']))
            
            # Group into rows
            rows = []
            current_row = [regions[0]]
            current_y = regions[0]['y_center']
            y_threshold = max(10, regions[0]['height'] * 0.5)  # dynamic threshold
            
            for region in regions[1:]:
                if abs(region['y_center'] - current_y) < y_threshold:
                    current_row.append(region)
                else:
                    # Sort current row by X
                    current_row.sort(key=lambda r: r['x_center'])
                    rows.append(current_row)
                    current_row = [region]
                    current_y = region['y_center']
            
            # Append last row
            if current_row:
                current_row.sort(key=lambda r: r['x_center'])
                rows.append(current_row)
            
            # Extract table data
            table_data = []
            for row in rows:
                row_text = [region['text'] for region in row]
                table_data.append(row_text)
            
            # Save to Excel
            if table_data:
                # Pad rows to same length
                max_cols = max([len(r) for r in table_data]) if table_data else 1
                table_data = [r + [''] * (max_cols - len(r)) for r in table_data]
                
                df = pd.DataFrame(table_data)
                out_path = output_folder / f"{os.path.splitext(fname)[0]}_extracted.xlsx"
                df.to_excel(out_path, index=False, header=False)
                # Uncomment to see progress per file:
                # print(f"✅ Saved {out_path.name}")
        
        except Exception as e:
            print(f"❌ Error processing {fname}: {str(e)[:50]}")
    
    print(f"\n🎉 Done! Results saved in {output_folder}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fast Handwritten Table OCR')
    parser.add_argument('--input_folder', required=True, help='Input folder with table images')
    parser.add_argument('--output_folder', required=True, help='Output folder for Excel files')
    args = parser.parse_args()
    
    print("\n📊 Handwritten Table Detection & OCR Pipeline")
    print("=" * 50)
    process_table_images(
        args.input_folder,
        args.output_folder
    )
