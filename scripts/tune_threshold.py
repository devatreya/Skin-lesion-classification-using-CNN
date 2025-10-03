#!/usr/bin/env python3
"""
Threshold tuning utility for binary classification.
Finds optimal threshold for target sensitivity (e.g., 90% for clinical use).
"""

import sys
from pathlib import Path
import numpy as np
import json
import argparse
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def load_validation_probabilities(run_dir):
    """Load validation probabilities from saved file."""
    prob_path = Path(run_dir) / 'val_probs.npz'
    
    if not prob_path.exists():
        raise FileNotFoundError(
            f"Validation probabilities not found at {prob_path}\n"
            "Make sure training has completed and saved val_probs.npz"
        )
    
    data = np.load(prob_path)
    return data['y_true'], data['y_prob']


def compute_metrics_at_threshold(y_true, y_prob, threshold):
    """Compute all metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision / Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    
    return {
        'threshold': float(threshold),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv),
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def find_optimal_threshold(y_true, y_prob, target_sensitivity=0.90, num_points=1001):
    """
    Find optimal threshold that achieves target sensitivity.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_prob: Predicted probabilities
        target_sensitivity: Minimum sensitivity to achieve (e.g., 0.90)
        num_points: Number of thresholds to evaluate
    
    Returns:
        best_threshold: Threshold achieving target sensitivity
        best_metrics: Metrics at that threshold
        all_results: List of metrics at all thresholds
    """
    thresholds = np.linspace(0, 1, num_points)
    all_results = []
    
    best_threshold = 0.5
    best_metrics = None
    best_specificity = 0.0
    
    for t in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_prob, t)
        all_results.append(metrics)
        
        # Find threshold with target sensitivity and highest specificity
        if metrics['sensitivity'] >= target_sensitivity:
            if metrics['specificity'] > best_specificity:
                best_threshold = t
                best_metrics = metrics
                best_specificity = metrics['specificity']
    
    # If target not achievable, find closest
    if best_metrics is None:
        print(f"⚠️ Target sensitivity {target_sensitivity:.2f} not achievable.")
        print("   Finding threshold with highest sensitivity...")
        max_sens = max(r['sensitivity'] for r in all_results)
        for r in all_results:
            if r['sensitivity'] == max_sens:
                best_threshold = r['threshold']
                best_metrics = r
                break
    
    return best_threshold, best_metrics, all_results


def plot_threshold_curves(y_true, y_prob, all_results, best_metrics, output_dir):
    """Plot sensitivity/specificity vs threshold curves."""
    thresholds = [r['threshold'] for r in all_results]
    sensitivities = [r['sensitivity'] for r in all_results]
    specificities = [r['specificity'] for r in all_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Sensitivity & Specificity vs Threshold
    ax1.plot(thresholds, sensitivities, label='Sensitivity', linewidth=2, color='#e74c3c')
    ax1.plot(thresholds, specificities, label='Specificity', linewidth=2, color='#3498db')
    ax1.axvline(best_metrics['threshold'], color='green', linestyle='--', alpha=0.7,
                label=f"Optimal: {best_metrics['threshold']:.3f}")
    ax1.axhline(best_metrics['sensitivity'], color='red', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Sensitivity & Specificity vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ax2.plot(fpr, tpr, linewidth=2, color='#e74c3c',
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    
    # Mark operating point
    y_pred_best = (y_prob >= best_metrics['threshold']).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_best).ravel()
    fpr_best = fp / (fp + tn)
    tpr_best = tp / (tp + fn)
    ax2.plot(fpr_best, tpr_best, 'go', markersize=10,
             label=f"Operating point (Sens={tpr_best:.2f}, 1-Spec={fpr_best:.2f})")
    
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax2.set_title('ROC Curve with Operating Point', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'threshold_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Threshold analysis plot saved to {plot_path}")
    plt.show()


def print_operating_point_table(all_results, target_sensitivities=[0.80, 0.85, 0.90, 0.95]):
    """Print table of operating points for different target sensitivities."""
    print(f"\n{'='*80}")
    print("OPERATING POINTS FOR DIFFERENT SENSITIVITY TARGETS")
    print(f"{'='*80}")
    print(f"{'Sensitivity':<12} {'Threshold':<12} {'Specificity':<12} {'PPV':<12} {'NPV':<12} {'F1':<10}")
    print(f"{'-'*80}")
    
    for target_sens in target_sensitivities:
        best_spec = 0
        best_row = None
        
        for r in all_results:
            if r['sensitivity'] >= target_sens:
                if r['specificity'] > best_spec:
                    best_spec = r['specificity']
                    best_row = r
        
        if best_row:
            print(f"{best_row['sensitivity']:.4f}       "
                  f"{best_row['threshold']:.4f}       "
                  f"{best_row['specificity']:.4f}       "
                  f"{best_row['ppv']:.4f}       "
                  f"{best_row['npv']:.4f}       "
                  f"{best_row['f1_score']:.4f}")
        else:
            print(f"{target_sens:.4f}       NOT ACHIEVABLE")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Tune classification threshold for target sensitivity')
    parser.add_argument('--run_dir', type=str, default='outputs',
                       help='Directory containing val_probs.npz')
    parser.add_argument('--target_sensitivity', type=float, default=0.90,
                       help='Target sensitivity (recall) for malignant class (default: 0.90)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (defaults to run_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.run_dir
    
    print(f"\n{'='*80}")
    print("THRESHOLD TUNING FOR SKIN LESION CLASSIFICATION")
    print(f"{'='*80}\n")
    
    # Load validation probabilities
    print(f"Loading validation probabilities from {args.run_dir}...")
    try:
        y_true, y_prob = load_validation_probabilities(args.run_dir)
        print(f"✅ Loaded {len(y_true)} validation samples")
        print(f"   Benign: {(y_true == 0).sum()} | Malignant: {(y_true == 1).sum()}")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("\nTo generate val_probs.npz, modify your training script to save:")
        print("  np.savez('val_probs.npz', y_true=y_true, y_prob=y_prob)")
        return
    
    # Find optimal threshold
    print(f"\nFinding optimal threshold for {args.target_sensitivity:.1%} sensitivity...")
    best_threshold, best_metrics, all_results = find_optimal_threshold(
        y_true, y_prob, target_sensitivity=args.target_sensitivity
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("OPTIMAL THRESHOLD FOUND")
    print(f"{'='*80}")
    print(f"Threshold: {best_metrics['threshold']:.4f}")
    print(f"Sensitivity (Recall): {best_metrics['sensitivity']:.4f}")
    print(f"Specificity: {best_metrics['specificity']:.4f}")
    print(f"PPV (Precision): {best_metrics['ppv']:.4f}")
    print(f"NPV: {best_metrics['npv']:.4f}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"F1 Score: {best_metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {best_metrics['tn']:4d}  |  FP: {best_metrics['fp']:4d}")
    print(f"  FN: {best_metrics['fn']:4d}  |  TP: {best_metrics['tp']:4d}")
    print(f"{'='*80}\n")
    
    # Print operating point table
    print_operating_point_table(all_results)
    
    # Save threshold to JSON
    threshold_path = Path(args.output_dir) / 'threshold.json'
    with open(threshold_path, 'w') as f:
        json.dump(best_metrics, f, indent=2)
    print(f"✅ Threshold saved to {threshold_path}")
    
    # Plot curves
    plot_threshold_curves(y_true, y_prob, all_results, best_metrics, args.output_dir)
    
    print(f"\n{'='*80}")
    print("THRESHOLD TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Use threshold {best_metrics['threshold']:.4f} for inference")
    print(f"2. Evaluate on test set ONCE with this threshold")
    print(f"3. Generate Grad-CAM visualizations for error analysis")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

