import os
from pathlib import Path
import cv2
import pprint
import pandas as pd

os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
from paddleocr import PaddleOCR

p=Path('workspace/ss thesis')
imgs=[f for f in p.glob('*') if f.suffix.lower() in ['.jpg','.png','.jpeg','.bmp']]
if not imgs:
    print('No images found')
    raise SystemExit(1)

img_path=str(imgs[0])
print('Image:',img_path)
ocr=PaddleOCR(use_textline_orientation=False, lang='en')
res=ocr.ocr(img_path, det=True, rec=True)
print('len res:', len(res))
print('len res[0]:', len(res[0]))

# Build raw_boxes as in src
raw_boxes=[]
for line in res[0]:
    if isinstance(line, (list, tuple)) and len(line) >= 1:
        box = line[0] if isinstance(line[0], list) else line
        raw_boxes.append([box, 0])

print('raw_boxes count:', len(raw_boxes))

# group
import importlib.util
spec = importlib.util.spec_from_file_location('paddle_ocr_mod', 'src/paddle_ocr.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rows = module.group_boxes_into_rows(raw_boxes, y_threshold=20)
print('rows count:', len(rows))
if rows:
    print('first row len:', len(rows[0]))

# Recognize fallback
img_cv = cv2.imread(img_path)
table_data=[]
for row_boxes in rows:
    row_text=[]
    for box_pack in row_boxes:
        box = box_pack[0]
        x_coords=[int(p[0]) for p in box]
        y_coords=[int(p[1]) for p in box]
        x_min, x_max = max(0, min(x_coords)), max(x_coords)
        y_min, y_max = max(0, min(y_coords)), max(y_coords)
        pad=5
        h,w=img_cv.shape[:2]
        y_min=max(0,y_min-pad)
        y_max=min(h,y_max+pad)
        x_min=max(0,x_min-pad)
        x_max=min(w,x_max+pad)
        crop=img_cv[y_min:y_max, x_min:x_max]
        print('crop size:', crop.shape if hasattr(crop,'shape') else None)
        # write crop
        crop_path=f'scripts/crop_debug_{len(row_text)}.png'
        cv2.imwrite(crop_path, crop)
        rec = ocr.ocr(crop_path, det=False, rec=True)
        print('rec for crop:', rec)
        text=''
        if rec and len(rec)>0:
            first=rec[0]
            if isinstance(first, (list,tuple)) and len(first)>0 and isinstance(first[0],(tuple,list)) and isinstance(first[0][0],str):
                text=first[0][0]
            else:
                # try deeper
                found=None
                for item in first:
                    if isinstance(item,(list,tuple)):
                        if len(item)>=2 and isinstance(item[-1],(tuple,list)) and isinstance(item[-1][0],str):
                            found=item[-1][0]
                            break
                        if isinstance(item[0],str):
                            found=item[0]
                            break
                if found:
                    text=found
        row_text.append(text)
    table_data.append(row_text)

print('table_data examples:')
for i,r in enumerate(table_data[:5]):
    print(i, r[:10])

# Save to excel
if table_data:
    max_cols = max([len(r) for r in table_data])
    table_data = [r + [''] * (max_cols - len(r)) for r in table_data]
    df=pd.DataFrame(table_data)
    df.to_excel('outputs/no_trocr/test_out.xlsx', index=False, header=False)
    print('wrote outputs/no_trocr/test_out.xlsx')
else:
    print('no table_data to save')
