#!/usr/bin/env pwsh
<#
Full pipeline runner for Windows PowerShell.
Adjust variables below as required, then run from project root:
    .\run_all_training.ps1
#>

# Activate venv
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "Virtual environment activation script not found at .\.venv\Scripts\Activate.ps1. Ensure venv exists or activate manually." -ForegroundColor Yellow
}

# --- Parameters (can be passed from command line) ---
param(
    [string] $WORK_DIR = (Resolve-Path .\workspace),
    [string] $IMAGES_DIR = "",
    [string] $LABELS_XLSX = "",
    [int] $EPOCHS_MAIN = 50,
    [int] $BATCH_MAIN = 32,
    [int] $EPOCHS_EMNIST = 25,
    [int] $BATCH_EMNIST = 64
)

# Default images/labels inside workspace if not provided explicitly
if (-not $IMAGES_DIR -or $IMAGES_DIR -eq "") {
    $IMAGES_DIR = "$(Resolve-Path $WORK_DIR)\custom\Images"
}
if (-not $LABELS_XLSX -or $LABELS_XLSX -eq "") {
    $LABELS_XLSX = "$(Resolve-Path $WORK_DIR)\custom\mydatasetexcel.xlsx"
}

Write-Host "Work dir: $WORK_DIR"
Write-Host "Images dir: $IMAGES_DIR"
Write-Host "Labels xlsx: $LABELS_XLSX"

# 1) Prepare and train your custom model
if (Test-Path $IMAGES_DIR -PathType Container -ErrorAction SilentlyContinue) {
    Write-Host "Preparing dataset from $IMAGES_DIR and labels $LABELS_XLSX"
    python run_train.py --images_dir $IMAGES_DIR --labels $LABELS_XLSX --work_dir $WORK_DIR --epochs $EPOCHS_MAIN --batch $BATCH_MAIN --augment
} else {
    Write-Host "Using existing prepared dataset at $WORK_DIR\dataset_augmented"
    python run_train.py --work_dir $WORK_DIR --epochs $EPOCHS_MAIN --batch $BATCH_MAIN
}

# 2) Run EMNIST benchmark
Write-Host "Running EMNIST benchmark..."
python run_emnist.py --emnist_dir "$WORK_DIR\emnist_yolo" --work_dir $WORK_DIR --epochs $EPOCHS_EMNIST --batch $BATCH_EMNIST

# 3) Plot training results
Write-Host "Plotting training curves (if available)..."
python run_plot_training.py $WORK_DIR\runs

Write-Host "All steps finished. Check $WORK_DIR\runs and $WORK_DIR\checkpoints for artifacts." -ForegroundColor Green
