#!/usr/bin/env python3
"""
src/evaluate.py
----------------
Model Evaluation & Performance Benchmarking for Deepfake and AI-Generated Image Detection.

Evaluates trained model weights against the unseen test split (`data/deepfake_dataset/test/`)
and outputs:
  - Accuracy, Precision, Recall, F1-Score, and AUC-ROC
  - Confusion Matrix plot (outputs/evaluation/test_set/confusion_matrix.png)
  - Receiver Operating Characteristic (ROC) curve (outputs/evaluation/test_set/roc_curve.png)
  - Latency benchmarks (ms per frame and FPS)
  - Comprehensive metrics summary JSON & text reports
"""

import argparse
import json
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)
import torch
from ultralytics import YOLO

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detector on unseen test set.")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/deepfake_detector/weights/best.pt",
        help="Path to trained classification weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/deepfake_dataset",
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image resolution")
    parser.add_argument("--project", type=str, default="outputs/evaluation", help="Evaluation output directory")
    parser.add_argument("--name", type=str, default="test_set", help="Subdirectory for evaluation artifacts")
    return parser.parse_args()


def collect_test_dataset(split_dir: Path):
    """
    Collect all test images and ground truth binary labels.
    Label mapping:
      0: Real (authentic)
      1: Fake (deepfake / AI-generated)
    """
    samples = []
    labels = []

    real_dir = split_dir / "real"
    fake_dir = split_dir / "fake"

    if real_dir.is_dir():
        for p in real_dir.iterdir():
            if p.suffix.lower() in VALID_EXTENSIONS:
                samples.append(p)
                labels.append(0)

    if fake_dir.is_dir():
        for p in fake_dir.iterdir():
            if p.suffix.lower() in VALID_EXTENSIONS:
                samples.append(p)
                labels.append(1)

    return samples, np.array(labels)


def main():
    args = parse_arguments()

    model_path = Path(args.model)
    data_path = Path(args.data)
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        # Check last.pt fallback
        last_path = model_path.parent / "last.pt"
        if last_path.exists():
            print(f"Notice: '{model_path}' not found, using latest checkpoint: '{last_path}'")
            model_path = last_path
        else:
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'.\n"
                "Please run 'src/train.py' first to train the detector."
            )

    split_dir = data_path / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Test split directory not found at: {split_dir}")

    print("\n" + "=" * 65)
    print("      DEEPFAKE & AI IMAGE DETECTION: PERFORMANCE EVALUATION")
    print("=" * 65)
    print(f"Model path:     {model_path.resolve()}")
    print(f"Evaluating:     {split_dir.resolve()}")
    print(f"Resolution:     {args.imgsz}x{args.imgsz}")
    print("=" * 65 + "\n")

    # Load model
    model = YOLO(str(model_path.resolve()))
    model_classes = model.names  # e.g., {0: 'fake', 1: 'real'} or {0: 'real', 1: 'fake'}
    print(f"Model class mapping: {model_classes}")

    # Determine which index corresponds to 'fake'
    fake_class_idx = 1
    for idx, name in model_classes.items():
        if "fake" in name.lower():
            fake_class_idx = idx
            break

    # Gather test samples
    samples, y_true = collect_test_dataset(split_dir)
    if len(samples) == 0:
        raise RuntimeError(f"No test images found in '{split_dir}'. Ensure real/ and fake/ subfolders contain images.")

    print(f"Collected {len(samples)} images for evaluation (Real: {np.sum(y_true == 0)}, Fake: {np.sum(y_true == 1)}).")

    # Run batched inference & latency benchmark
    y_pred = []
    y_scores = []
    latencies = []

    for img_path in samples:
        t0 = time.perf_counter()
        results = model.predict(source=str(img_path), imgsz=args.imgsz, verbose=False)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        res = results[0]
        top1_idx = int(res.probs.top1)
        probs = res.probs.data.cpu().numpy()

        # Score is probability of being FAKE
        fake_prob = float(probs[fake_class_idx])
        pred_label = 1 if (top1_idx == fake_class_idx) else 0

        y_pred.append(pred_label)
        y_scores.append(fake_prob)

    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_fake = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_fake = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    precision_real = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_real = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_real = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    # ROC AUC
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = float(auc(fpr, tpr))
    except Exception:
        fpr, tpr = [0, 1], [0, 1]
        roc_auc = 0.50

    # Latency metrics
    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake/AI"],
        yticklabels=["Real", "Fake/AI"],
    )
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.1%})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Plot ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1e88e5", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Recall)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = out_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # Build report text
    report = f"""
============================================================
       DEEPFAKE & AI IMAGE DETECTION EVALUATION REPORT
============================================================
Evaluated Split:     {args.split} ({len(samples)} total images)
Model Weights:       {model_path.name}
Resolution:          {args.imgsz}x{args.imgsz}

Overall Performance:
  * Accuracy:             {accuracy:.4f}  ({accuracy * 100:.2f}%)
  * ROC AUC Score:        {roc_auc:.4f}  ({roc_auc * 100:.2f}%)
  * Average Latency:      {avg_latency:.2f} ms per image
  * Inference Speed:      {fps:.1f} FPS

Fake / AI-Generated Class Metrics (Target):
  * Precision:            {precision_fake:.4f}  ({precision_fake * 100:.2f}%)
  * Recall:               {recall_fake:.4f}  ({recall_fake * 100:.2f}%)
  * F1-Score:             {f1_fake:.4f}

Real / Authentic Class Metrics:
  * Precision:            {precision_real:.4f}  ({precision_real * 100:.2f}%)
  * Recall:               {recall_real:.4f}  ({recall_real * 100:.2f}%)
  * F1-Score:             {f1_real:.4f}

Confusion Matrix Breakdown:
  * True Negatives (Real correctly identified):   {tn}
  * False Positives (Real falsely flagged Fake):  {fp}
  * False Negatives (Fake missed / slipped past): {fn}
  * True Positives (Fake correctly detected):     {tp}

Artifacts Generated:
  * Confusion Matrix:  {cm_path.resolve()}
  * ROC Curve:         {roc_path.resolve()}
============================================================
"""
    print(report)

    # Save summary files
    summary_txt = out_dir / "metrics_summary.txt"
    summary_txt.write_text(report, encoding="utf-8")

    metrics_json = out_dir / "metrics_summary.json"
    metrics_data = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "precision_fake": float(precision_fake),
        "recall_fake": float(recall_fake),
        "f1_fake": float(f1_fake),
        "precision_real": float(precision_real),
        "recall_real": float(recall_real),
        "f1_real": float(f1_real),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "latency_ms": float(avg_latency),
        "fps": float(fps),
    }
    metrics_json.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    print(f"Summary metrics saved to:\n  - {summary_txt}\n  - {metrics_json}")


if __name__ == "__main__":
    main()
