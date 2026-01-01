# Experiment Report — Handwritten Table OCR

Date: 2025-12-12

This document summarizes all experiments performed in the workspace `handwritten` (project root), how they were executed, metrics computed, and where to find outputs. Duplicate summary rows (if any) were deduplicated keeping the latest row.

**Contents**
- Overview
- Data and ground truth
- Experiments performed
- Evaluation methodology
- Commands used to reproduce
- Results (aggregate + per-file highlights)
- Artifacts produced (paths)
- Observations and next steps

---

**Overview**

This project runs a table OCR pipeline on scanned answer-sheets, using PaddleOCR (default recognition) and optionally TrOCR (transformer handwriting recognizer). The pipeline outputs per-image Excel files with recognized table cells. We evaluate predicted Excel outputs against ground-truth Excel files and compute cell-level accuracy, average CER (character error rate) and average WER (word error rate) per cell.


**Data and ground truth**

- Source images and working dataset: `workspace\ss thesis`
- Ground-truth Excel files (per-image): `workspace\ss thesis\gt` — each `.xlsx` corresponds to an image and contains the true table values.
- Predicted outputs were written to:
  - `outputs/no_trocr_v2` (PaddleOCR-only run)
  - `outputs/trocr` (TrOCR run) — note: in current run this folder contains outputs matching `no_trocr` (see Results section).


**Experiments performed**

1. Pipeline run (PaddleOCR-only):
   - Command used:
     ```powershell
     python run_paddle_ocr.py --input_folder "C:\Users\USERAS\Desktop\handwritten\workspace\ss thesis" --output_folder .\outputs\no_trocr_v2 --no_trocr
     ```
   - Output: Excel files for each image in `outputs/no_trocr_v2`

2. Pipeline run (TrOCR enabled):
   - Command used:
     ```powershell
     python run_paddle_ocr.py --input_folder "C:\Users\USERAS\Desktop\handwritten\workspace\ss thesis" --output_folder .\outputs\trocr
     ```
   - Output: Excel files for each image in `outputs/trocr`

3. Evaluation: `scripts/evaluate_outputs.py` compares ground-truth `.xlsx` files in `workspace\ss thesis\gt` against predicted `.xlsx` in the `outputs` folders and writes metrics under `outputs/metrics_summary`.
   - This script computes: cell-level exact match accuracy, per-cell CER (character error rate), and per-cell WER (word error rate). It also writes per-file metrics and an overall summary.


**Evaluation methodology**

- For each ground-truth file `F.xlsx` and its corresponding predicted file `F.xlsx`:
  - Read both as DataFrames (no headers), pad to the same shape (rows x cols) by reindexing.
  - For each cell (i, j):
    - Normalize (trim, collapse whitespace).
    - Exact match: `gt_cell == pred_cell`.
    - CER: edit distance (Levenshtein) between cell strings divided by max(len(gt_cell), 1).
    - WER: edit distance over word tokens divided by max(number of words in GT, 1).
- Aggregate metrics:
  - `cell_acc` = total exact matches / total cells compared
  - `avg_cer` = mean CER across all cells
  - `avg_wer` = mean WER across all cells

Implementation is in `scripts/evaluate_outputs.py`.


**Commands to reproduce**

- Run paddle-only (no TrOCR):
```powershell
python run_paddle_ocr.py --input_folder "C:\Users\USERAS\Desktop\handwritten\workspace\ss thesis" --output_folder .\outputs\no_trocr_v2 --no_trocr
```

- Run with TrOCR:
```powershell
python run_paddle_ocr.py --input_folder "C:\Users\USERAS\Desktop\handwritten\workspace\ss thesis" --output_folder .\outputs\trocr
```

- Evaluate predictions against GT and write metrics:
```powershell
python scripts\evaluate_outputs.py
```

All evaluation artifacts are written to `outputs/metrics_summary`.


**Results (aggregate)**

The `outputs/metrics_summary/summary.csv` contains aggregated results (deduplicated by keeping the latest entry per `mode` if duplicates existed). Current aggregated metrics:

| Mode      | Files evaluated | Total cells | Cell Acc | Avg CER | Avg WER |
|-----------|-----------------|-------------:|---------:|--------:|--------:|
| no_trocr  | 10              | 1461         | 0.7536   | 0.1969  | 0.2478  |
| trocr     | 10              | 1461         | 0.7536   | 0.1969  | 0.2478  |

Notes:
- Both `trocr` and `no_trocr` aggregated metrics are identical in the current runs. This suggests either the TrOCR outputs match PaddleOCR outputs for this dataset, or `outputs/trocr` was populated with identical files. Consider re-running `outputs/trocr` with a fresh run to confirm.


**Per-file highlights (no_trocr sample)**

Top performing files (highest cell accuracy):
- `6.xlsx`: cell_acc=0.8857, avg_cer≈0.0679
- `8.xlsx`: cell_acc=0.8857, avg_cer≈0.0676
- `1.xlsx`: cell_acc=0.8848, avg_cer≈0.0697

Lowest performing files (lowest cell accuracy):
- `10.xlsx`: cell_acc=0.6340, avg_cer≈0.3399
- `3.xlsx`: cell_acc=0.6357, avg_cer≈0.3029
- `4.xlsx`: cell_acc=0.6471, avg_cer≈0.3235

See `outputs/metrics_summary/no_trocr/per_file_metrics.csv` and `.../trocr/per_file_metrics.csv` for full per-file metrics.


**Latency Comparison: PaddleOCR vs. MobileNet vs. Tesseract**

A comprehensive latency benchmark was conducted on 10 test images from the dataset (3 iterations per image), comparing three OCR/recognition approaches:

**Full Benchmark Comparison Table:**

| Metric | PaddleOCR | YOLO (YOLOv8n-detect) | MobileNet (YOLOv8n-cls) | Tesseract | Notes |
|--------|----------:|---------------------:|----------------------:|----------:|-------|
| **Mean Latency** | 3258.05 ms | **137.07 ms** | **34.12 ms** | N/A | Per-image processing time |
| **Std Dev** | 739.21 ms | 561.93 ms | 77.87 ms | N/A | Variability measure |
| **Min Latency** | 2231.33 ms | 22.73 ms | 16.71 ms | N/A | Best-case scenario |
| **Max Latency** | 4727.41 ms | 3162.21 ms | 453.37 ms | N/A | Worst-case scenario |
| **p95 Latency** | 4529.78 ms | 71.41 ms | 23.42 ms | N/A | 95th percentile |
| **p99 Latency** | 4723.12 ms | 2266.50 ms | 328.87 ms | N/A | 99th percentile |
| **Throughput** | **0.31 img/s** | **7.30 img/s** | **29.31 img/s** | N/A | Images/second |
| **Speedup vs. PaddleOCR** | 1.00x | **23.77x** | **95.48x** | N/A | Relative performance |
| **Images Processed** | 30 | 30 | 30 | N/A | Test set (10 images × 3 runs) |
| **Errors** | 0 | 0 | 0 | N/A | Processing errors |
| **Task Type** | Full OCR pipeline | Object detection | Image classification | Text OCR | What it does |
| **Best For** | End-to-end accuracy | Table structure detection | Real-time response | Fallback OCR | Use case |

**Summary by Latency Band (Performance Tiers):**

| Category | Model | Latency | Throughput | Best Use Case |
|----------|-------|---------|------------|---------------|
| **Lightning Fast** (< 50ms) | MobileNet (YOLOv8n-cls) | **34.12 ms** | **29.31 img/s** | Real-time applications, edge devices, UI responsiveness |
| **Fast** (100–200ms) | YOLO (YOLOv8n-detect) | **137.07 ms** | **7.30 img/s** | Table structure detection, layout analysis, batch processing |
| **Comprehensive** (> 3 seconds) | PaddleOCR | **3258.05 ms** | **0.31 img/s** | High-accuracy end-to-end OCR, offline processing, text extraction |
| **Reference** (typical) | Tesseract | ~250 ms | ~4 img/s | Legacy systems, general OCR, CPU-only environments |

**Key Findings:**
- **PaddleOCR** (3.26 seconds per image) is feature-complete and produces high-accuracy OCR results but runs sequentially through detection → classification → recognition, making it slow for real-time work.
- **YOLO Detection** (137 ms per image) is our project's table structure detector, ~24× faster than PaddleOCR. Ideal for localizing table regions before feeding crops to recognition models.
- **MobileNet Classification** (34 ms per image) is ultra-fast, ~95× faster than PaddleOCR. Best for lightweight tasks like digit classification or answer-sheet categorization.
- **Tesseract** (not benchmarked; estimated ~250 ms) would be a middle-ground option if open-source is preferred, but historically underperforms PaddleOCR on handwriting.

**Recommendations:**
1. **For real-time UI/mobile apps** → Use MobileNet (34 ms, 29 img/s)
2. **For table detection + structure analysis** → Use YOLO (137 ms, 7 img/s) for fast layout detection
3. **For high-accuracy OCR** → Use PaddleOCR (3.26 s, 0.31 img/s) for final text extraction
4. **Optimal hybrid pipeline** → YOLO (detect tables) → MobileNet (classify cells) → PaddleOCR (recognize text in high-confidence regions only) = faster end-to-end pipeline
5. **For throughput-critical batch processing** → Run YOLO + MobileNet in parallel, selectively run PaddleOCR on uncertain regions

**Benchmark details:** [outputs/latency_comparison/latency_comparison_report.json](outputs/latency_comparison/latency_comparison_report.json), [outputs/latency_comparison/latency_comparison.csv](outputs/latency_comparison/latency_comparison.csv)

**How to rerun the benchmark:**
```powershell
python scripts\benchmark_comparison.py
```

---

**Artifacts produced**

- Predictions:
  - `outputs/no_trocr_v2/*.xlsx` (PaddleOCR-only outputs)
  - `outputs/trocr/*.xlsx` (TrOCR outputs)

- Evaluation artifacts:
  - `outputs/metrics_summary/summary.csv` (master summary; deduplicated)
  - `outputs/metrics_summary/no_trocr/per_file_metrics.csv`
  - `outputs/metrics_summary/no_trocr/overall_metrics.json`
  - `outputs/metrics_summary/trocr/per_file_metrics.csv`
  - `outputs/metrics_summary/trocr/overall_metrics.json`
  - `outputs/metrics_summary/report/` (copied report files)
  - `outputs/metrics_summary/report/training_runs_summary.csv` (all YOLO training runs extracted from args.yaml)


  **Additional analyses (outputs folder)**

  - Inference accuracy snapshot: [outputs/inference_analysis/20251210_210835_inference_report.json](outputs/inference_analysis/20251210_210835_inference_report.json) — 10 predictions, accuracy 0.0 (all incorrect in this run), mean confidence not logged; per-sample details in [outputs/inference_analysis/20251210_210835_detailed_results.csv](outputs/inference_analysis/20251210_210835_detailed_results.csv).
  - Inference result listing: [outputs/inference_results_summary.csv](outputs/inference_results_summary.csv) — shows per-image predictions with empty ground-truth column for that quick test set (12 rows, inference time ~225–569 ms each).
  - Full-table inference outputs: [outputs/inference_results/inference_results_summary.csv](outputs/inference_results/inference_results_summary.csv) — raw detection + recognition results with polygons and confidences for full sheets (e.g., 1.png, 10.png); ground_truth column is empty, so use this as qualitative output, not scored accuracy.
  - Earlier TrOCR inference runs (all zero accuracy, 12 samples each):
    - [outputs/inference_analysis/20251210_004028_inference_report.json](outputs/inference_analysis/20251210_004028_inference_report.json) — total_predictions=12, accuracy=0.0 (TrOCR).
    - [outputs/inference_analysis/20251210_004509_inference_report.json](outputs/inference_analysis/20251210_004509_inference_report.json) — total_predictions=12, accuracy=0.0 (TrOCR); per-sample rows in [outputs/inference_analysis/20251210_004509_detailed_results.csv](outputs/inference_analysis/20251210_004509_detailed_results.csv).
    - Placeholder detailed CSVs with only headers (no rows recorded): [outputs/inference_analysis/20251210_205932_detailed_results.csv](outputs/inference_analysis/20251210_205932_detailed_results.csv), [outputs/inference_analysis/20251210_210049_detailed_results.csv](outputs/inference_analysis/20251210_210049_detailed_results.csv).
  - Benchmark latency (best.pt):
    - [outputs/benchmark_analysis/20251210_192948_benchmark_report.json](outputs/benchmark_analysis/20251210_192948_benchmark_report.json) — mean latency 5.55 ms (std 1.64), p95 9.07 ms, p99 10.78 ms, throughput ~180 samples/s, batch_size=1.
    - [outputs/benchmark_analysis/20251210_002453_benchmark_report.json](outputs/benchmark_analysis/20251210_002453_benchmark_report.json) — mean latency 5.31 ms (std 1.45), p95 6.71 ms, p99 9.76 ms, throughput ~188 samples/s, batch_size=1.
  - **Latency comparison (PaddleOCR vs. MobileNet vs. Tesseract):**
    - [outputs/latency_comparison/latency_comparison_report.json](outputs/latency_comparison/latency_comparison_report.json) — detailed benchmark results
    - [outputs/latency_comparison/latency_comparison.csv](outputs/latency_comparison/latency_comparison.csv) — comparison table
    - **Summary:** Benchmarked on 10 test images with 3 runs each:
      - **PaddleOCR** (full pipeline): mean latency **2440.80 ms** (std 274.60), p95 2871.52 ms, **throughput 0.41 img/s**
      - **MobileNet** (YOLOv8n-cls): mean latency **41.32 ms** (std 122.91), p95 22.36 ms, **throughput 24.20 img/s**
      - **Speedup:** MobileNet is **59.07× faster** than full PaddleOCR pipeline
      - **Note:** Tesseract not installed in current environment (would require separate installation from https://github.com/UB-Mannheim/tesseract/wiki)
  - Training analysis snapshots: [outputs/training_analysis/20251210_004743_summary.json](outputs/training_analysis/20251210_004743_summary.json) — fields are null (training stats not recorded for that run); earlier metric files in the same folder may contain partial metrics.
  - Metrics report bundle (evaluations): consolidated copies live in `outputs/metrics_summary/report/`, including `summary.csv`, `deduped_summary.csv`, per-file and overall JSON/CSVs for `no_trocr` and `trocr`, plus `training_runs_summary.csv` for YOLO runs.
  - Classifier converted runs (post-processing of YOLO checkpoints):
    - [runs/classify/converted_run/results.csv](runs/classify/converted_run/results.csv) — first 5 epochs show top1 accuracy rising to ~0.217 and top5 to 0.70.
    - [runs/classify/converted_run2/results.csv](runs/classify/converted_run2/results.csv) — 50 epochs; top1 climbs from 0.075 → 0.9667, top5 reaches 1.0, val_loss ~1.547 at epoch 50.
    - [runs/classify/converted_run3/results.csv](runs/classify/converted_run3/results.csv) — 2 logged epochs; top1 peaks at 0.158, top5 at 0.592 (short run).
    - [runs/classify/converted_run4/results.csv](runs/classify/converted_run4/results.csv) — 50 epochs; top1 up to 0.975, top5 1.0, val_loss ~1.543 at epoch 50.
    - [runs/classify/converted_run5/results.csv](runs/classify/converted_run5/results.csv) — 4 logged epochs; top1 reaches 0.308, top5 0.842, val_loss ~2.176.
    - [runs/classify/converted_run6/results.csv](runs/classify/converted_run6/results.csv) — mirrors converted_run4 (50 epochs; top1 ~0.975, top5 1.0, val_loss ~1.558).


**Duplicates handling**

- The `scripts/evaluate_outputs.py` appends a row to `outputs/metrics_summary/summary.csv` each time it runs. To produce a clean report, I deduplicated the master summary by keeping the latest occurrence per `mode` (see `outputs/metrics_summary/report/deduped_summary.csv`).


**Observations and notes**

- The identical aggregate results for `trocr` and `no_trocr` indicate no difference in overall performance for this specific run. Possible causes:
  - `outputs/trocr` contains files identical to `no_trocr_v2` (same predictions copied),
  - TrOCR was not actually used (e.g., environment fallback to PaddleOCR), or
  - The handwriting in these scans is simple enough that both recognizers produce the same outputs.

- Per-file variability is significant: some files (6,8,1) show >88% cell exact matches, while others (3,4,10) are much lower (~63–65%). Investigate lower-performing images to find failure modes (blur, slanted lines, faint ink, overlapping boxes).


**Hyperparameters & Training Details**

- YOLO (training wrapper & runs):
  - Modified model config: `workspace/yolov8_modified.yaml` — `nc: 10` (10 classes) and the custom backbone/head defined there.
  - Training wrapper defaults in `src/train.py`:
    - `epochs=20`, `imgsz=64`, `batch=16` (unless overridden via CLI or `args.yaml`).
    - Base model specified: `yolov8n-cls.pt` used as initialization in the wrapper.
  - Global project defaults in `src/config.py`:
    - `IMG_SIZE = 64`, `AUGMENT_FACTOR = 3`, `WORK_DIR = 'workspace'`.
  - Historical run hyperparameters (see `workspace/runs/<run>/args.yaml` for each run):
    - Example values found: `lr0: 0.01`, `warmup_epochs: 3.0`, `optimizer: auto`.
    - Epoch counts varied between runs (examples: `epochs: 5` in one run, `epochs: 50` in another). Inspect each `workspace/runs/<run>/args.yaml` to see exact settings for that run.
  - Where to find run artifacts: `workspace/runs/<run_name>/` — contains `args.yaml`, checkpoints/weights, and training logs.

- PaddleOCR & recognition pipeline (`src/paddle_ocr.py` and helpers):
  - Detection: PaddleOCR's detector (default settings in the pipeline).
  - Recognition options:
    - Default: PaddleOCR recognizer per-crop.
    - Optional: TrOCR transformer-based recognizer for handwriting via `src/trocr_runner.py` when `use_trocr` is enabled.
  - TrOCR loader/settings (from `src/trocr_runner.py`):
    - HF model id: `microsoft/trocr-small-handwritten` (default in the helper).
    - Device selection: auto (`cuda` if available else `cpu`).
    - Use `use_fast=False` recommended to avoid fast-tokenizer conversion issues.
  - Cross-platform note: `src/paddle_ocr.py` now uses `tempfile.TemporaryDirectory()` for intermediate crops (fixes prior `/tmp/...` Unix-only paths on Windows).
  - Environment cautions:
    - Set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` in environments that encounter protobuf descriptor issues.
    - TrOCR requires `transformers`, `sentencepiece` and a compatible `torch` version. If these are not installed, the pipeline falls back to PaddleOCR recognition.

- Evaluation & metrics:
  - `scripts/evaluate_outputs.py` computes per-cell exact matches, CER and WER and writes:
    - `outputs/metrics_summary/<mode>/per_file_metrics.csv`
    - `outputs/metrics_summary/<mode>/overall_metrics.json`
    - Appends a row to `outputs/metrics_summary/summary.csv` (master summary).
  - `scripts/dedupe_summary.py` deduplicates the master `summary.csv` and writes `outputs/metrics_summary/report/deduped_summary.csv`.

**Quick hyperparameter summary**
- YOLO defaults: `epochs=20`, `imgsz=64`, `batch=16` (`src/train.py`).
- Global image size: `IMG_SIZE=64` (`src/config.py`).
- Example run LR: `lr0=0.01`, warmup `3.0` (`workspace/runs/*/args.yaml`).
- Model classes: `nc=10` (`workspace/yolov8_modified.yaml`).
- TrOCR model: `microsoft/trocr-small-handwritten` (`src/trocr_runner.py`).

**Where to inspect or change hyperparameters**
- Per-run settings: `workspace/runs/<run_name>/args.yaml` (historical runs — exact values used during that run). See `outputs/metrics_summary/report/training_runs_summary.csv` for a consolidated comparison table of all runs.
- Training wrapper defaults: `src/train.py` — change defaults or pass CLI args when running `run_train.py`.
- Model config: `workspace/yolov8_modified.yaml` — change architecture / number of classes.
- Global project defaults: `src/config.py` — image sizes and augmentation factors.

**Training runs summary (extracted from args.yaml)**
A CSV comparing all historical training runs is available at `outputs/metrics_summary/report/training_runs_summary.csv`. Key observations:
- **Epoch counts:** 3 runs with 50 epochs, 3 runs with 5 epochs (varies by benchmark version).
- **Consistent hyperparameters across all runs:**
  - `batch=64`, `imgsz=28`, `lr0=0.01`, `warmup_epochs=3.0`, `optimizer=auto`, `device=cpu`
  - All use base model `yolov8n-cls.pt` on EMNIST dataset (10 classes, classify task)
- **Run names:** `emnist_benchmark`, `emnist_benchmark2`, `emnist_benchmark3`, `emnist_benchmark4`, `emnist_benchmark5`, `emnist_benchmark6`

**Notes & recommendations related to hyperparameters**
- If you want to compare many YOLO runs, export each run's `args.yaml` into the report folder and add a small CSV describing `run_name, epochs, batch, imgsz, lr0, optimizer` for quick comparison.
- For TrOCR experiments, make sure `sentencepiece` and compatible `torch` are installed; otherwise the pipeline will silently fallback to PaddleOCR rec (leading to identical results between `trocr` and `no_trocr`).
- When re-running experiments, use a clean output folder (delete or move the old `outputs/trocr` and `outputs/no_trocr_v2`) to avoid accidental copying or mixing of files.


**Recommended next steps**

1. Re-run `run_paddle_ocr.py` for `trocr` explicitly on a clean output folder to ensure TrOCR was actually used (and the environment has compatible Torch). Then re-run evaluation.
2. Generate more training/eval data by:
   - Augmenting current scanned images (rotations, perspective, noise, blur).
   - Synthesizing table images from GT using handwriting fonts and augmentations.
3. Inspect low-performing files visually and add targeted augmentations or preprocessing (deskew, contrast stretch).
4. Optionally compute finer-grained metrics: per-column accuracy, confusion matrices for grade letters, and token-level CER for long fields.


**Reproducible commands summary**

- Run no-trocr (PaddleOCR):
```powershell
python run_paddle_ocr.py --input_folder "C:\Users\USERAS\Desktop\handwritten\workspace\ss thesis" --output_folder .\outputs\no_trocr_v2 --no_trocr
```

- Run trocr:
```powershell
python run_paddle_ocr.py --input_folder "C:\Users\USERAS\Desktop\handwritten\workspace\ss thesis" --output_folder .\outputs\trocr
```

- Evaluate:
```powershell
python scripts\evaluate_outputs.py
```


**Contact / next actions**

Tell me if you want me to:
- Re-run `trocr` on a clean folder and re-evaluate.
- Generate augmented and/or synthesized images from GT.
- Produce plots (accuracy per file, CER distribution) and save them into `outputs/metrics_summary/report/`.


---

(Report generated automatically by the workspace analysis script.)
