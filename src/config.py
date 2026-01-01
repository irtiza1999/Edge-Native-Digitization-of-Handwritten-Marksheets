import os
from pathlib import Path

# Default configuration; override via function args or env vars where appropriate
BASE_DIR = Path(os.getenv('HW_BASE_DIR', '.')).absolute()
WORK_DIR = Path(os.getenv('HW_WORK_DIR', BASE_DIR / 'workspace')).absolute()
DATASET_ROOT = WORK_DIR / 'dataset_augmented'
IMG_SIZE = int(os.getenv('HW_IMG_SIZE', 64))
AUGMENT_FACTOR = int(os.getenv('HW_AUG_FACTOR', 3))

# Checkpoints
CHECKPOINT_DIR = WORK_DIR / 'checkpoints'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(DATASET_ROOT, exist_ok=True)

cfg = {
    'BASE_DIR': str(BASE_DIR),
    'WORK_DIR': str(WORK_DIR),
    'DATASET_ROOT': str(DATASET_ROOT),
    'IMG_SIZE': IMG_SIZE,
    'AUGMENT_FACTOR': AUGMENT_FACTOR,
}
