"""Visualization and analysis utilities for training, inference, and benchmarking."""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class TrainingAnalytics:
    """Log and visualize training metrics."""
    
    def __init__(self, output_dir: str = "outputs/training_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'time_per_epoch': [],
            'timestamp': []
        }
        self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None,
                  lr: float = None, time_per_epoch: float = None):
        """Log metrics for a single epoch."""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        if lr is not None:
            self.metrics['learning_rate'].append(lr)
        if time_per_epoch is not None:
            self.metrics['time_per_epoch'].append(time_per_epoch)
        self.metrics['timestamp'].append(datetime.now().isoformat())
    
    def save_metrics_json(self):
        """Save metrics to JSON for later analysis."""
        json_path = self.output_dir / f"{self.run_name}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {json_path}")
    
    def plot_training_curves(self):
        """Plot loss and learning rate curves."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        axes[0].plot(self.metrics['epochs'], self.metrics['train_loss'], 
                    marker='o', label='Train Loss', linewidth=2, markersize=4)
        if self.metrics['val_loss']:
            axes[0].plot(self.metrics['epochs'], self.metrics['val_loss'], 
                        marker='s', label='Val Loss', linewidth=2, markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate curve (if available)
        if self.metrics['learning_rate']:
            axes[1].plot(self.metrics['epochs'], self.metrics['learning_rate'],
                        marker='^', color='green', linewidth=2, markersize=4)
            axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
            axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')
        else:
            # Plot time per epoch if no LR data
            if self.metrics['time_per_epoch']:
                axes[1].bar(self.metrics['epochs'], self.metrics['time_per_epoch'],
                           color='orange', alpha=0.7)
                axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
                axes[1].set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
                axes[1].set_title('Time per Epoch', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = self.output_dir / f"{self.run_name}_training_curves.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {path}")
    
    def plot_epoch_timing(self):
        """Plot epoch timing analysis."""
        if not self.metrics['time_per_epoch']:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        times = self.metrics['time_per_epoch']
        ax.plot(self.metrics['epochs'], times, marker='o', linewidth=2, markersize=6, color='steelblue')
        ax.fill_between(self.metrics['epochs'], times, alpha=0.3, color='steelblue')
        
        avg_time = np.mean(times)
        ax.axhline(y=avg_time, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_time:.2f}s')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.output_dir / f"{self.run_name}_epoch_timing.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Epoch timing saved to {path}")
    
    def generate_training_summary(self):
        """Generate text summary of training statistics."""
        summary = {
            'total_epochs': len(self.metrics['epochs']),
            'best_train_loss': min(self.metrics['train_loss']) if self.metrics['train_loss'] else None,
            'best_val_loss': min(self.metrics['val_loss']) if self.metrics['val_loss'] else None,
            'final_train_loss': self.metrics['train_loss'][-1] if self.metrics['train_loss'] else None,
            'final_val_loss': self.metrics['val_loss'][-1] if self.metrics['val_loss'] else None,
            'avg_time_per_epoch': np.mean(self.metrics['time_per_epoch']) if self.metrics['time_per_epoch'] else None,
            'total_training_time': sum(self.metrics['time_per_epoch']) if self.metrics['time_per_epoch'] else None,
        }
        
        summary_path = self.output_dir / f"{self.run_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nTraining Summary:")
        print("=" * 50)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key:25s}: {value:10.4f}")
            else:
                print(f"  {key:25s}: {value}")
        
        return summary


class InferenceAnalytics:
    """Analyze and visualize inference results."""
    
    def __init__(self, output_dir: str = "outputs/inference_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.predictions = []
        self.ground_truth = []
        self.confidences = []
        self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_prediction(self, pred_text: str, gt_text: str, confidence: float = None,
                      image_path: str = None, model_name: str = None):
        """Add a prediction result."""
        self.results.append({
            'prediction': pred_text,
            'ground_truth': gt_text,
            'confidence': confidence,
            'image_path': image_path,
            'model_name': model_name,
            'correct': pred_text.lower() == gt_text.lower()
        })
        self.predictions.append(pred_text.lower())
        self.ground_truth.append(gt_text.lower())
        if confidence is not None:
            self.confidences.append(confidence)
    
    def plot_confidence_distribution(self):
        """Plot distribution of prediction confidences."""
        if not self.confidences:
            print("Warning: no confidence scores available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(self.confidences, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(self.confidences), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(self.confidences):.3f}')
        ax.axvline(np.median(self.confidences), color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(self.confidences):.3f}')
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = self.output_dir / f"{self.run_name}_confidence_dist.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confidence distribution saved to {path}")
    
    def plot_accuracy_by_confidence(self):
        """Plot accuracy vs confidence threshold."""
        if not self.confidences or not self.results:
            print("Warning: not enough data for accuracy analysis")
            return
        
        confidence_thresholds = np.linspace(0, 1, 21)
        accuracies = []
        counts = []
        
        for threshold in confidence_thresholds:
            mask = [c >= threshold for c in self.confidences]
            if not any(mask):
                continue
            
            filtered_correct = sum([self.results[i]['correct'] for i in range(len(mask)) if mask[i]])
            filtered_count = sum(mask)
            accuracy = filtered_correct / filtered_count if filtered_count > 0 else 0
            accuracies.append(accuracy)
            counts.append(filtered_count)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy vs threshold
        ax1.plot(confidence_thresholds[:len(accuracies)], accuracies, marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy vs Confidence Threshold', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Count vs threshold
        ax2.bar(confidence_thresholds[:len(counts)], counts, color='orange', alpha=0.7)
        ax2.set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Predictions Retained', fontsize=12, fontweight='bold')
        ax2.set_title('Predictions vs Confidence Threshold', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = self.output_dir / f"{self.run_name}_accuracy_by_confidence.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Accuracy analysis saved to {path}")
    
    def generate_inference_report(self):
        """Generate detailed inference analysis report."""
        if not self.results:
            print("Warning: no inference results to analyze")
            return
        
        correct = sum([r['correct'] for r in self.results])
        total = len(self.results)
        accuracy = correct / total if total > 0 else 0
        
        report = {
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'incorrect_predictions': total - correct,
            'mean_confidence': float(np.mean(self.confidences)) if self.confidences else None,
            'median_confidence': float(np.median(self.confidences)) if self.confidences else None,
            'std_confidence': float(np.std(self.confidences)) if self.confidences else None,
        }
        
        # Count by model (if available)
        model_names = set([r['model_name'] for r in self.results if r['model_name']])
        if model_names:
            report['by_model'] = {}
            for model in model_names:
                model_results = [r for r in self.results if r['model_name'] == model]
                model_correct = sum([r['correct'] for r in model_results])
                report['by_model'][model] = {
                    'count': len(model_results),
                    'correct': model_correct,
                    'accuracy': model_correct / len(model_results) if model_results else 0
                }
        
        report_path = self.output_dir / f"{self.run_name}_inference_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nInference Report:")
        print("=" * 50)
        print(f"  Total Predictions: {report['total_predictions']}")
        print(f"  Correct: {report['correct_predictions']}")
        print(f"  Accuracy: {report['accuracy']:.2%}")
        if report['mean_confidence'] is not None:
            print(f"  Mean Confidence: {report['mean_confidence']:.4f}")
        if 'by_model' in report:
            print(f"\n  By Model:")
            for model, stats in report['by_model'].items():
                print(f"    {model}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['count']})")
        
        return report
    
    def save_detailed_results_csv(self):
        """Save detailed results to CSV for inspection."""
        import csv
        
        csv_path = self.output_dir / f"{self.run_name}_detailed_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['prediction', 'ground_truth', 'correct', 'confidence', 'model_name', 'image_path'])
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"Detailed results saved to {csv_path}")


class BenchmarkAnalytics:
    """Analyze and visualize benchmark results."""
    
    def __init__(self, output_dir: str = "outputs/benchmark_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timings = {}
        self.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_timing(self, model_name: str, latencies: List[float], throughput: float = None,
                  batch_size: int = 1):
        """Add timing data for a model."""
        self.timings[model_name] = {
            'latencies': latencies,
            'throughput': throughput,
            'batch_size': batch_size,
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
        }
    
    def plot_latency_distribution(self):
        """Plot latency distributions for all models."""
        if not self.timings:
            print("Warning: no timing data available")
            return
        
        fig, axes = plt.subplots(1, len(self.timings), figsize=(6 * len(self.timings), 5))
        if len(self.timings) == 1:
            axes = [axes]
        
        for ax, (model_name, data) in zip(axes, self.timings.items()):
            latencies = data['latencies']
            ax.hist(latencies, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(data['mean_latency'], color='red', linestyle='--', linewidth=2,
                      label=f"Mean: {data['mean_latency']:.3f}ms")
            ax.axvline(np.median(latencies), color='orange', linestyle='--', linewidth=2,
                      label=f"Median: {np.median(latencies):.3f}ms")
            ax.set_xlabel('Latency (ms)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title(f'{model_name}\n(n={len(latencies)})', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = self.output_dir / f"{self.run_name}_latency_distribution.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Latency distribution saved to {path}")
    
    def plot_model_comparison(self):
        """Plot latency comparison across models."""
        if not self.timings:
            return
        
        model_names = list(self.timings.keys())
        means = [self.timings[m]['mean_latency'] for m in model_names]
        stds = [self.timings[m]['std_latency'] for m in model_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(model_names))
        ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.set_ylabel('Mean Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Model Latency Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = self.output_dir / f"{self.run_name}_model_comparison.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model comparison saved to {path}")
    
    def generate_benchmark_report(self):
        """Generate benchmark report."""
        if not self.timings:
            print("Warning: no benchmark data available")
            return
        
        report = {}
        for model_name, data in self.timings.items():
            report[model_name] = {
                'mean_latency_ms': float(data['mean_latency']),
                'std_latency_ms': float(data['std_latency']),
                'min_latency_ms': float(data['min_latency']),
                'max_latency_ms': float(data['max_latency']),
                'p95_latency_ms': float(data['p95_latency']),
                'p99_latency_ms': float(data['p99_latency']),
                'throughput': data['throughput'],
                'batch_size': data['batch_size'],
            }
        
        report_path = self.output_dir / f"{self.run_name}_benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nBenchmark Report:")
        print("=" * 70)
        for model_name, stats in report.items():
            print(f"\n{model_name}:")
            print(f"  Mean Latency:    {stats['mean_latency_ms']:10.3f} ms")
            print(f"  Std Dev:         {stats['std_latency_ms']:10.3f} ms")
            print(f"  Min Latency:     {stats['min_latency_ms']:10.3f} ms")
            print(f"  Max Latency:     {stats['max_latency_ms']:10.3f} ms")
            print(f"  P95 Latency:     {stats['p95_latency_ms']:10.3f} ms")
            print(f"  P99 Latency:     {stats['p99_latency_ms']:10.3f} ms")
            if stats['throughput']:
                print(f"  Throughput:      {stats['throughput']:10.2f} samples/sec")
        
        return report
