# Handwritten Recognition Project

This repository is a converted and modularized version of `handwritten_main_v2.ipynb`.
It provides data preparation, YOLO-based classification training, and TrOCR inference, with GPU support.

Quick start (Windows PowerShell):

1) Create a virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies. For CUDA-enabled PyTorch, follow the official instructions at https://pytorch.org/get-started/locally and install the matching `torch`, `torchvision`, `torchaudio` package before installing the rest. Then:

```powershell
pip install -r requirements.txt
```

**Important: Protobuf Compatibility (for PaddleOCR)**

If you encounter a `TypeError: Descriptors cannot be created directly` error when using PaddleOCR (e.g., in `run_paddle_ocr.py`), it's due to a protobuf version mismatch. Choose one fix:

**Option A (Recommended):** Pin protobuf to a compatible version:
```powershell
pip install "protobuf<=3.20.3"
```

**Option B (Runtime workaround, if Option A is not available):** Set the environment variable:
```powershell
# For current session only:
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'

# For permanent use (all future sessions), run in PowerShell as Administrator:
setx PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
```

Note: Option B uses pure-Python protobuf parsing (slower but avoids the error). Option A is preferred for better performance.

3) Prepare data and run training (example):

```powershell
python run_train.py --images_dir "C:\path\to\Images" --labels "C:\path\to\labels.xlsx" --work_dir .\workspace
```

4) Run TrOCR inference on folder:

```powershell
python run_inference.py --input_folder "C:\path\to\input_images" --output_folder ./outputs
```

Notes:
- The code will use GPU if available (PyTorch detects CUDA). Ensure you installed a CUDA-enabled `torch` build.
- `requirements.txt` includes CPU fallbacks. If you need GPU-enabled packages, install `torch` with the correct CUDA version first.

Using a local `workspace` dataset
- If you placed your raw dataset under `workspace/custom` or `workspace/ss thesis`, the script will auto-discover it and build `workspace/dataset_augmented` if needed.
- To test discovery without running training, use `--dry_run`:

```powershell
python run_train.py --work_dir .\workspace --dry_run
```

After a successful training run, checkpoints are copied to `workspace/checkpoints/converted_run_<timestamp>/`.

Checkpoints and saving weights
- Training copies the produced `best.pt` and `last.pt` (if present) from the ultralytics `runs` folder into `workspace/checkpoints/converted_run_<timestamp>/` so you can load them later.
- To load a saved model later, point the YOLO loader at the `.pt` file:

```python
from ultralytics import YOLO
model = YOLO('workspace/checkpoints/converted_run_YYYY.../best.pt')
```

Quick steps if your dataset is already prepared in `workspace`:

```powershell
# If you already copied a prepared dataset structure into workspace/dataset_augmented (with train/val folders):
python run_train.py --work_dir .\workspace --epochs 5 --batch 32
```

Available Scripts and Modules

**Training & Checkpoints**
- `python run_train.py --work_dir .\workspace --epochs 20 --batch 32`
  Trains YOLO on your custom dataset, saves checkpoints to `workspace/checkpoints/`.
- `python run_train.py --work_dir .\workspace --dry_run`
  Tests dataset discovery without starting training.

**Visualization**
- `python run_plot_training.py ./workspace/runs`
  Plots training curves (accuracy, loss) and shows confusion matrices from the latest run.

**EMNIST Benchmark (Standard Dataset)**
- `python run_emnist.py --emnist_dir .\workspace\emnist_yolo --work_dir .\workspace --epochs 25 --batch 64`
  Downloads EMNIST digits, trains on them, and benchmarks against your custom dataset.
- `python run_emnist.py --emnist_dir .\workspace\emnist_yolo --work_dir .\workspace --custom_dataset .\workspace\dataset_augmented --eval_only`
  Evaluates a saved EMNIST model on your custom dataset (cross-dataset generalization).

**CPU Latency Benchmark**
- `python scripts/benchmark_cpu.py --model .\workspace\checkpoints\converted_run_YYYY\best.pt`
  Measures CPU inference latency (useful for edge device deployment).

**Table Detection & Text Recognition (Handwritten Tables)**
- `python run_paddle_ocr.py --input_folder ".\path\to\images" --output_folder .\outputs`
  Detects text boxes in images using PaddleOCR and recognizes each cell with TrOCR, saves results to Excel.
- `python run_paddle_ocr.py --input_folder ".\path\to\images" --output_folder .\outputs --no_trocr`
  Use PaddleOCR for recognition instead of TrOCR.

**TrOCR Handwriting Recognition**
- `python run_inference.py --input_folder ".\path\to\images" --output_folder .\outputs`
  Runs TrOCR on single images and saves recognized text to Excel files.

Full Training Guide (one-shot)
------------------------------
This section shows exactly how to run the full training pipeline on your local machine (Windows PowerShell). The example script `run_all_training.ps1` included will:

- Prepare the dataset (if you provide `--images_dir` and `--labels`) or use the prepared dataset under `workspace/dataset_augmented`.
- Train your custom YOLO classifier using the converted notebook code.
- Run the EMNIST benchmark training.
- Produce checkpoints under `workspace/checkpoints/` and runs under `workspace/runs/`.

1) Edit variables at the top of the PowerShell script if you want to customize epochs, batch sizes, or paths.

2) Run the script in PowerShell (from the project root). It will activate the venv, install missing deps if requested, and run the steps sequentially.

PowerShell one-shot example (run locally):

```powershell
# Run the full pipeline script (from project root)
powershell -ExecutionPolicy Bypass -File .\run_all_training.ps1
```

Contents of `run_all_training.ps1` (the script in this repo does the same):

```powershell
# Activate virtual environment (adjust path if different)
.\.venv\Scripts\Activate.ps1

# Configuration
$WORK_DIR = Resolve-Path .\workspace
$IMAGES_DIR = "$WORK_DIR\custom\Images"        # optional: raw images folder
$LABELS_XLSX = "$WORK_DIR\custom\mydatasetexcel.xlsx" # optional: labels file
$EPOCHS_MAIN = 50
$BATCH_MAIN = 32
$EPOCHS_EMNIST = 25
$BATCH_EMNIST = 64

Write-Host "Starting full training pipeline..."

# 1) Prepare and train your custom model (build dataset if images+labels provided)
if (Test-Path $IMAGES_DIR -PathType Container -ErrorAction SilentlyContinue) {
  Write-Host "Preparing dataset from $IMAGES_DIR and labels $LABELS_XLSX"
  python run_train.py --images_dir $IMAGES_DIR --labels $LABELS_XLSX --work_dir $WORK_DIR --epochs $EPOCHS_MAIN --batch $BATCH_MAIN --augment
} else {
  Write-Host "Using existing prepared dataset at $WORK_DIR\dataset_augmented"
  python run_train.py --work_dir $WORK_DIR --epochs $EPOCHS_MAIN --batch $BATCH_MAIN
}

# 2) Run EMNIST benchmark (downloads EMNIST, converts, trains)
Write-Host "Running EMNIST benchmark..."
python run_emnist.py --emnist_dir "$WORK_DIR\emnist_yolo" --work_dir $WORK_DIR --epochs $EPOCHS_EMNIST --batch $BATCH_EMNIST

# 3) Optional: plot results
Write-Host "Plotting training curves (if results available)..."
python run_plot_training.py $WORK_DIR\runs

Write-Host "Full training pipeline complete. Check $WORK_DIR\runs and $WORK_DIR\checkpoints for outputs."
```

## 📊 Analysis and Visualization

All pipelines (training, inference, and benchmarking) now include comprehensive analytics and visualizations automatically generated during execution.

### Training Analysis

After running training with `run_train.py`, analytics are automatically saved to `outputs/training_analysis/`:

```
outputs/training_analysis/
├── YYYYMMDD_HHMMSS_training_curves.png     # Loss and learning rate curves
├── YYYYMMDD_HHMMSS_epoch_timing.png        # Time per epoch analysis
├── YYYYMMDD_HHMMSS_metrics.json            # Raw metrics data
└── YYYYMMDD_HHMMSS_summary.json            # Training summary statistics
```

**Example**: To train and see visualizations:
```powershell
python run_train.py --images_dir "C:\path\to\Images" --labels "C:\path\to\labels.xlsx" --work_dir .\workspace
# Check outputs/training_analysis/ for training curves and timing plots
```

### Inference Analysis

Run inference with analysis:

```powershell
python run_inference.py --input_folder ./sample_images --output_folder ./outputs --ground_truth_folder ./groundtruth_texts
```

Output: `outputs/inference_analysis/`

```
outputs/inference_analysis/
├── YYYYMMDD_HHMMSS_confidence_dist.png              # Confidence score distribution
├── YYYYMMDD_HHMMSS_accuracy_by_confidence.png       # Accuracy vs confidence threshold
├── YYYYMMDD_HHMMSS_inference_report.json            # Detailed accuracy metrics
├── YYYYMMDD_HHMMSS_detailed_results.csv             # Per-image results
└── YYYYMMDD_HHMMSS_metrics.json                     # Raw metrics
```

**Note**: If you provide `--ground_truth_folder` with `.txt` files (one text per image), the analyzer will compute accuracy statistics.

### Benchmark Analysis

CPU latency benchmarking with automatic visualizations:

```powershell
python scripts/benchmark_cpu.py --model .\workspace\checkpoints\converted_run_*\best.pt --iterations 100
```

Output: `outputs/benchmark_analysis/`

```
outputs/benchmark_analysis/
├── YYYYMMDD_HHMMSS_latency_distribution.png    # Latency histogram
├── YYYYMMDD_HHMMSS_model_comparison.png        # Model comparison bars
├── YYYYMMDD_HHMMSS_benchmark_report.json       # Detailed latency statistics
│   ├── mean_latency_ms
│   ├── std_latency_ms
│   ├── p95_latency_ms
│   ├── p99_latency_ms
│   └── throughput (FPS)
└── YYYYMMDD_HHMMSS_metrics.json
```

### Table OCR Analysis

Extract handwritten table data with automatic detection:

```powershell
python run_table_ocr.py --input_folder "C:\Users\USERAS\Desktop\handwritten\workspace\ss thesis" --output_folder .\outputs
```

Output: `outputs/*.xlsx` files with extracted text organized into rows and columns.

---

## 📁 Project Structure

```
.
├── src/
│   ├── config.py              # Configuration (paths, hyperparameters)
│   ├── data.py                # Data preparation and augmentation
│   ├── train.py               # YOLO training wrapper with analytics
│   ├── trocr_runner.py        # TrOCR inference wrapper
│   ├── paddle_ocr_fast.py     # Fast handwritten table detection
│   └── analytics.py           # Visualization and analysis utilities (NEW)
├── scripts/
│   ├── benchmark_cpu.py       # CPU latency benchmark with analysis
│   └── download_emnist.py     # EMNIST dataset downloader
├── run_train.py               # Training entrypoint
├── run_inference.py           # TrOCR inference entrypoint
├── run_emnist.py              # EMNIST benchmark entrypoint
├── run_table_ocr.py           # Table OCR entrypoint
├── run_all_training.ps1       # PowerShell orchestration script
└── outputs/                   # All analysis outputs
    ├── training_analysis/     # Training curves and summaries
    ├── inference_analysis/    # Inference accuracy and confidence analysis
    └── benchmark_analysis/    # Latency and throughput reports
```

Notes & Tips
- The script is sequential and will run on whatever device your Python environment uses (GPU if available). Each training step may take long depending on epochs and dataset size.
- If you want to run experiments in parallel or on separate GPUs, split the steps into separate terminals and adjust `--device` flags accordingly.
- After each training run, the script copies the observed `best.pt` and `last.pt` to `workspace/checkpoints/converted_run_<timestamp>/` for easy retrieval.
- **All visualizations are automatically saved to `outputs/` folders. Open them with any image viewer or directly from the outputs folder.**

