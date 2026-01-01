"""Handwritten table detection and OCR runner using improved cell detection."""
import argparse
from src.paddle_ocr_fast import process_table_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Handwritten Table OCR: detect table cells and extract text to Excel')
    parser.add_argument('--input_folder', required=True, help='Input folder with handwritten table images')
    parser.add_argument('--output_folder', required=True, help='Output folder for Excel files')
    args = parser.parse_args()
    
    import os
    from pathlib import Path
    
    print("\n📊 Handwritten Table Detection & OCR Pipeline")
    print("=" * 50)
    
    # If the provided input folder doesn't exist or contains no images, try auto-discovery
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"Warning: input folder '{args.input_folder}' does not exist. Attempting to auto-discover image folders...")
        # Search common workspace folders for images and pick the folder with most images
        candidates = []
        search_roots = [Path('.'), Path('workspace'), Path('workspace') / 'dataset_augmented', Path('workspace') / 'custom', Path('workspace') / 'ss thesis']
        for root in search_roots:
            if not root.exists():
                continue
            for p in root.rglob('*'):
                if p.is_dir():
                    count = len([f for f in p.glob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
                    if count > 0:
                        candidates.append((count, p))
        
        if candidates:
            # pick the directory with the most images
            candidates.sort(reverse=True)
            best = candidates[0][1]
            print(f"Auto-discovered input folder: {best} ({candidates[0][0]} images)")
            args.input_folder = str(best)
        else:
            print("No image folders found under common locations. Please create or point to a folder with images and re-run.")
            raise SystemExit(1)
    
    # Call the processing function
    process_table_images(
        args.input_folder,
        args.output_folder
    )
