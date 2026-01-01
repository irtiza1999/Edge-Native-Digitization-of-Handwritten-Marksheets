import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from paddleocr import PaddleOCR
import cv2

img = r"C:\Users\USERAS\Desktop\handwritten\workspace\ss thesis\1.png"
print('Loading PaddleOCR...')
ocr = PaddleOCR(use_textline_orientation=False, lang='en')
print('PaddleOCR loaded')

print('Calling predict/ocr on image...')
# Try predict and ocr if available
if hasattr(ocr, 'predict'):
    try:
        p = ocr.predict([[img]])
        print('predict -> type:', type(p))
        print('predict repr:', repr(p)[:1000])
    except Exception as e:
        print('predict error:', e)

if hasattr(ocr, 'ocr'):
    try:
        o = ocr.ocr(img, det=True, rec=True)
        print('ocr -> type:', type(o))
        print('ocr repr:', repr(o)[:1000])
    except Exception as e:
        print('ocr error:', e)

# If predict returned nested, print first element detail
try:
    res = p if 'p' in locals() else o
    print('Top-level length:', len(res))
    if len(res) > 0:
        print('Type of res[0]:', type(res[0]))
        print('repr(res[0])[:500]:', repr(res[0])[:500])
except Exception as e:
    print('Examining result failed:', e)

# If ocr returns object-like, try to extract text fields
try:
    if isinstance(res, list) and len(res) and hasattr(res[0], '__iter__'):
        print('Sample detection count:', len(res[0]))
        if len(res[0])>0:
            print('Sample element 0 type:', type(res[0][0]))
            print('Sample element 0 repr[:200]:', repr(res[0][0])[:200])
except Exception as e:
    print('Extraction failed:', e)

# Try using ocr on crop
img_cv = cv2.imread(img)
if img_cv is None:
    print('Could not load image')
else:
    h,w = img_cv.shape[:2]
    crop = img_cv[0:min(100,h), 0:min(200,w)]
    cv2.imwrite('debug_crop.png', crop)
    try:
        r_crop = ocr.ocr('debug_crop.png', det=False, rec=True)
        print('ocr on crop type:', type(r_crop))
        print('ocr on crop repr:', repr(r_crop)[:1000])
    except Exception as e:
        print('ocr on crop error:', e)
print('Done')
