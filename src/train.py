#!/usr/bin/env python3
"""
src/train.py
-------------
Deep Learning Model Training for Deepfake and AI-Generated Image Detection.

Fine-tunes a Deep Convolutional Neural Network (YOLOv8-Cls / EfficientNet / ResNet)
on the balanced Real vs. Fake image dataset.

Outputs:
  - Best model checkpoint: outputs/deepfake_detector/weights/best.pt
  - Epoch-by-epoch loss & accuracy logs: outputs/deepfake_detector/results.csv
  - Training visualizations: outputs/deepfake_detector/results.png
"""

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Deepfake & AI Generated Image Classifier.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/deepfake_dataset",
        help="Path to dataset root containing train/ and val/ folders",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-cls.pt",
        help="Pretrained classifier checkpoint (e.g., yolov8n-cls.pt, yolov8s-cls.pt)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size (default: 224)")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (default: 8)")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers (default: 2)")
    parser.add_argument("--device", type=str, default="", help="Device: '0' for CUDA GPU, 'cpu' for CPU")
    parser.add_argument("--project", type=str, default="outputs", help="Output directory")
    parser.add_argument("--name", type=str, default="deepfake_detector", help="Experiment run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_arguments()

    data_path = Path(args.data)
    output_path = Path(args.project)

    # Validate dataset existence
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            "Please run 'src/prepare_dataset.py' first to organize or generate Real vs Fake images."
        )

    # Verify classes inside train
    classes = [d.name.lower() for d in train_dir.iterdir() if d.is_dir()]
    print(f"Detected Classes in {train_dir}: {classes}")
    if len(classes) < 2:
        raise ValueError(
            f"Dataset must contain at least 2 classes ('real' and 'fake'). Found: {classes}"
        )

    # Determine execution device
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = 0
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Device] Using GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        else:
            device = "cpu"
            print("[Device] CUDA not available. Running on CPU.")

    print("\n" + "=" * 65)
    print("      DEEPFAKE & AI GENERATED IMAGE DETECTION: TRAINING")
    print("=" * 65)
    print(f"Dataset root:      {data_path.resolve()}")
    print(f"Architecture:      {args.model}")
    print(f"Target classes:    Real vs. Fake / AI-Generated")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch size:        {args.batch}")
    print(f"Image resolution:  {args.imgsz}x{args.imgsz}")
    print(f"Output directory:  {output_path.resolve() / args.name}")
    print("=" * 65 + "\n")

    # Initialize Ultralytics classification model
    model = YOLO(args.model)

    # Train model
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        project=str(output_path.resolve()),
        name=args.name,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    best_weights = output_path / args.name / "weights" / "best.pt"
    print("\n" + "=" * 65)
    print("Training Complete!")
    if best_weights.is_file():
        print(f"Optimal Model Checkpoint: {best_weights.resolve()}")
    print(f"Training Logs & Plots:    {(output_path / args.name).resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
