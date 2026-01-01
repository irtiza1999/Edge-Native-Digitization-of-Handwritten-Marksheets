"""CLI wrapper to prepare data and start training."""
import argparse
from src import config
from src.data import index_images, load_labels, process_dataset
from src.train import create_yolo_yaml, train_yolo
from pathlib import Path
import torch
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=False, help='Path to raw images folder. If omitted, assumes dataset already in work_dir/dataset_augmented')
    parser.add_argument('--labels', required=False, help='Path to labels excel. If omitted and images_dir provided, searches nearby. If dataset already prepared, omit.')
    parser.add_argument('--dry_run', action='store_true', help='If set, do not run training; only prepare dataset and print discovered paths')
    parser.add_argument('--work_dir', default=config.cfg['WORK_DIR'])
    parser.add_argument('--imgsz', type=int, default=config.cfg['IMG_SIZE'])
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    data_root = work_dir / 'dataset_augmented'

    # If user provided an images_dir, build dataset from raw images + labels
    if args.images_dir:
        print('Indexing images...')
        image_map, count = index_images(args.images_dir)
        print(f'Found {count} images')

        if not args.labels:
            raise ValueError('When providing --images_dir you must also provide --labels to build dataset')

        print('Loading labels...')
        try:
            df = load_labels(args.labels)
        except FileNotFoundError as e:
            print(f"Labels file not found: {e}. Trying to locate labels near images folder...")
            img_parent = Path(args.images_dir).parent
            candidates = list(img_parent.glob('*.xls*')) if img_parent.exists() else []
            if not candidates and img_parent.parent.exists():
                candidates = list(img_parent.parent.glob('*.xls*'))
            if candidates:
                print('Found label candidate:', candidates[0])
                df = load_labels(str(candidates[0]))
            else:
                raise

        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

        print('Processing train set...')
        process_dataset(train_df, image_map, data_root / 'train', img_size=args.imgsz, augment=args.augment, augment_factor=config.cfg['AUGMENT_FACTOR'])
        print('Processing val set...')
        process_dataset(val_df, image_map, data_root / 'val', img_size=args.imgsz, augment=False)
    else:
        # Assume dataset already exists in work_dir/dataset_augmented
        # If it's missing or empty, try to auto-discover raw dataset folders under work_dir
        if not (data_root / 'train').exists() or not (data_root / 'val').exists() or not any((data_root).iterdir()):
            print(f"No prepared dataset found in {data_root}. Attempting to auto-discover raw dataset under {work_dir}...")
            # Look for common folders used in the notebook: 'custom' and 'ss thesis'
            candidates = []
            for candidate_name in ['custom', 'ss thesis', 'ss_thesis', 'data']:
                cand = work_dir / candidate_name
                if cand.exists():
                    candidates.append(cand)

            # Also search for any folder under work_dir that contains an 'Images' subfolder or any excel file
            if not candidates:
                for p in work_dir.iterdir():
                    if p.is_dir():
                        if (p / 'Images').exists() or list(p.glob('*.xls*')):
                            candidates.append(p)

            if candidates:
                chosen = candidates[0]
                print(f"Auto-discovered dataset folder: {chosen}")
                # Try to find images and labels inside
                img_dir = chosen / 'Images'
                if not img_dir.exists():
                    # maybe images are directly inside chosen
                    img_dir = chosen

                # Find excel file
                excel_candidates = list(chosen.glob('*.xls*'))
                excel_path = excel_candidates[0] if excel_candidates else None

                if not excel_path:
                    # try parent
                    excel_candidates = list(chosen.glob('**/*.xls*'))
                    excel_path = excel_candidates[0] if excel_candidates else None

                if not excel_path:
                    print('No labels excel found in discovered folder; dataset build will be skipped unless you provide --labels')
                else:
                    print(f'Using labels file: {excel_path}')
                    print('Indexing images...')
                    image_map, count = index_images(str(img_dir))
                    print(f'Found {count} images in {img_dir}')
                    if count > 0 and excel_path:
                        from sklearn.model_selection import train_test_split
                        df = load_labels(str(excel_path))
                        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
                        print('Processing train set...')
                        process_dataset(train_df, image_map, data_root / 'train', img_size=args.imgsz, augment=args.augment, augment_factor=config.cfg['AUGMENT_FACTOR'])
                        print('Processing val set...')
                        process_dataset(val_df, image_map, data_root / 'val', img_size=args.imgsz, augment=False)
                    else:
                        raise FileNotFoundError(f"Auto-discovered folder {chosen} didn't contain images or labels. Provide --images_dir and --labels.")
            else:
                raise FileNotFoundError(f"No prepared dataset found in {data_root} and no candidate raw dataset found under {work_dir}.")

    # Create YAML and train
    num_classes = len([p for p in (data_root / 'train').iterdir() if p.is_dir()])
    yaml_path = create_yolo_yaml(num_classes, str(work_dir / 'yolov8_modified.yaml'), str(work_dir))

    device_arg = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device_arg}')
    print('Starting training...')
    # Ensure the ultralytics settings save_dir points to our work_dir/runs
    from ultralytics import settings
    settings.update({'save_dir': str(work_dir / 'runs')})

    if args.dry_run:
        print('Dry run requested; skipping actual training.')
        print(f'Dataset root: {data_root}')
        print(f'YAML path: {yaml_path}')
        print('You can run without --dry_run to start training.')
        return

    results, elapsed, weights = train_yolo(yaml_path, str(data_root), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device_arg, name='converted_run')
    print(f'Training finished in {elapsed:.2f}s')

    # Copy weights (best.pt / last.pt) to checkpoints
    from shutil import copy2
    from src.config import cfg
    ckpt_root = Path(cfg['WORK_DIR']) / 'checkpoints' if 'WORK_DIR' in cfg else work_dir / 'checkpoints'
    ckpt_root.mkdir(parents=True, exist_ok=True)
    run_ckpt_dir = ckpt_root / f"converted_run_{int(time.time())}"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if weights:
        for name_w, path_w in weights.items():
            try:
                copy2(path_w, run_ckpt_dir / name_w)
                print(f'Copied {name_w} -> {run_ckpt_dir / name_w}')
            except Exception as e:
                print(f'Could not copy {path_w}: {e}')
    else:
        print('No best/last weights found in runs; check the ultralytics run folder.')


if __name__ == '__main__':
    main()
