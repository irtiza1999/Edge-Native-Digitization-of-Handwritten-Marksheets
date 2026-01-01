"""Quick smoke test to ensure modules import and GPU detection works."""
from src import config
import torch

def main():
    print('Config WORK_DIR =', config.cfg['WORK_DIR'])
    print('Dataset root =', config.cfg['DATASET_ROOT'])
    print('Torch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print('CUDA device name:', torch.cuda.get_device_name(0))
        except Exception as e:
            print('Could not query device name:', e)

if __name__ == '__main__':
    main()
