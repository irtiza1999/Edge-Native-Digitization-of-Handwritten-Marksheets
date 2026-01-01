"""Data preparation utilities extracted from the notebook.
Functions:
- index_images(images_dir)
- load_labels(labels_file)
- process_dataset(df, image_map, save_root, img_size, augment, augment_factor)
"""
from pathlib import Path
import os
import cv2
import shutil
import albumentations as A
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import getpass


def index_images(images_dir: str):
    images_dir = Path(images_dir)
    # Resolve common Colab -> Windows Drive mappings
    images_dir = Path(_resolve_drive_path(str(images_dir)))
    image_map = {}
    count = 0
    if not images_dir.exists():
        # Try to find a folder named like the requested folder somewhere on the local drives
        possible_dirs = glob.glob(f"**/{images_dir.name}", recursive=True)
        if possible_dirs:
            images_dir = Path(possible_dirs[0])
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                key = Path(f).stem.lower().strip()
                image_map[key] = os.path.join(root, f)
                count += 1
    return image_map, count


def load_labels(labels_file: str):
    """Load labels from Excel file. Handle missing files by searching nearby.
    If a directory is provided, search for Excel files in that directory.
    """
    labels_path = Path(labels_file)
    
    # If the path is a directory, search for Excel files in it
    if labels_path.is_dir():
        candidates = list(labels_path.glob('*.xls*'))
        if candidates:
            labels_path = candidates[0]
            print(f"Found Excel file in directory: {labels_path.name}")
        else:
            raise FileNotFoundError(f"No Excel file found in directory '{labels_file}'")
    
    # If provided path doesn't exist, try to resolve Drive-style paths and search nearby
    if not labels_path.exists():
        labels_path = Path(_resolve_drive_path(str(labels_path)))

    if not labels_path.exists():
        # Search for any Excel file in the same directory as the labels path or its parent
        search_dir = labels_path.parent if labels_path.parent.exists() else Path('.').resolve()
        candidates = list(search_dir.glob('*.xls*'))
        if not candidates:
            # try parent directories upward a few levels
            for p in [search_dir, search_dir.parent, search_dir.parent.parent]:
                if p.exists():
                    candidates = list(p.glob('*.xls*'))
                    if candidates:
                        break
        if candidates:
            labels_path = candidates[0]
            print(f"Auto-found Excel file: {labels_path.name}")
        else:
            raise FileNotFoundError(f"Labels Excel file not found at '{labels_file}' or nearby folders.")

    df = pd.read_excel(labels_path, header=None)
    df.columns = ['filename', 'label']
    df['label'] = df['label'].astype(str).str.strip()
    df['filename'] = df['filename'].astype(str).str.strip()
    return df


def _resolve_drive_path(p: str) -> str:
    """Resolve common Colab `/content/drive/My Drive/...` paths to Windows Google Drive mounts.

    Attempts several likely local mount points used by Google Drive for Desktop.
    If no mapping found, returns the original path.
    """
    if not p:
        return p
    # If user passed Colab-style path, map to Windows common locations
    if p.startswith('/content/drive') or 'My Drive' in p:
        # extract relative path after 'My Drive'
        parts = p.split('My Drive')
        if len(parts) > 1:
            rel = parts[1].lstrip('/\\')
        else:
            rel = os.path.basename(p)

        user = getpass.getuser()
        candidates = [
            os.path.join('G:\\', 'My Drive', rel),
            os.path.join('G:\\', rel),
            os.path.join(os.path.expanduser('~'), 'Google Drive', 'My Drive', rel),
            os.path.join(os.path.expanduser('~'), 'Google Drive', rel),
            os.path.join(os.path.expanduser('~'), 'DriveFS', 'My Drive', rel),
        ]

        for c in candidates:
            if os.path.exists(c):
                return c

    return p


def process_dataset(dataframe, image_map, save_root, img_size=64, augment=False, augment_factor=3):
    save_path = Path(save_root)
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    transform = A.Compose([
        A.Rotate(limit=10, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.ISONoise(p=0.3),
        A.Resize(img_size, img_size)
    ])

    success_count = 0

    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        fname_raw = row['filename']
        label = row['label']

        fname_clean = os.path.basename(fname_raw)
        key = os.path.splitext(fname_clean)[0].lower().strip()

        if key in image_map:
            src_path = image_map[key]
        else:
            continue

        class_dir = save_path / label
        class_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(src_path)
        if image is None:
            continue

        base_name = key
        resized = cv2.resize(image, (img_size, img_size))
        cv2.imwrite(str(class_dir / f"{base_name}_orig.jpg"), resized)
        success_count += 1

        if augment:
            for i in range(augment_factor - 1):
                try:
                    augmented = transform(image=image)['image']
                    cv2.imwrite(str(class_dir / f"{base_name}_aug{i}.jpg"), augmented)
                except Exception:
                    pass

    if success_count == 0:
        raise RuntimeError(f"No images saved at {save_root}. Check filename matching.")

    return success_count
