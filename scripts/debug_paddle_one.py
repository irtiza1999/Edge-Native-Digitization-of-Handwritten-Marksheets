import os
from pathlib import Path
import cv2
import pprint

# Ensure protobuf workaround early
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')

try:
    from paddleocr import PaddleOCR
except Exception as e:
    print('Failed to import PaddleOCR:', e)
    raise

p=Path('workspace/ss thesis')
imgs=[f for f in p.glob('*') if f.suffix.lower() in ['.jpg','.png','.jpeg','.bmp']]
if not imgs:
    print('No images found in',p)
    raise SystemExit(1)

img=imgs[0]
print('Using image:',img)

ocr=PaddleOCR(use_textline_orientation=False, lang='en')
res=ocr.ocr(str(img), det=True, rec=True)
print('Type of result:', type(res))
print('len(result):', len(res))
pp=pprint.PrettyPrinter(indent=2)
pp.pprint(res[:3])

# print first detection entry structure
if res and len(res)>0:
    first = res[0]
    print('\nType first:',type(first),'len:',len(first))
    if isinstance(first, list) and first:
        entry = first[0]
        print('\nFirst entry type:',type(entry))
        pp.pprint(entry)

# Also show a small crop recognition
img_cv = cv2.imread(str(img))
# choose a central 50x20 crop
h,w=img_cv.shape[:2]
cy, cx = h//2, w//2
crop=img_cv[max(0,cy-25):min(h,cy+25), max(0,cx-10):min(w,cx+10)]
cv2.imwrite('scripts/debug_crop.png', crop)
print('\nSaved debug crop to scripts/debug_crop.png')
rec = ocr.ocr('scripts/debug_crop.png', det=False, rec=True)
print('Crop rec type:', type(rec))
pp.pprint(rec)
