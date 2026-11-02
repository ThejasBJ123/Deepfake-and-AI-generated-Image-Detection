#!/usr/bin/env python3
"""
src/predict_image.py
---------------------
Deepfake and AI-Generated Image Inference & Forensic Inspection CLI.

Analyzes individual images or folders using both Deep Learning classification
and Computer Vision forensics (Error Level Analysis & 2D-FFT Power Spectrum).

Outputs:
  - Classification Verdict: REAL vs. FAKE
  - Confidence Probability (e.g. 98.7% FAKE)
  - Multi-panel visual forensic proof saved to outputs/predictions/
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# Import local forensic suite
from forensics import compute_ela, compute_fft_spectrum, compute_noise_residual

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Deepfake & AI Image Detection on an image or directory.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file or directory")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/deepfake_detector/weights/best.pt",
        help="Path to trained classification weights",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Classification threshold for flagging FAKE (default: 0.50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions",
        help="Directory to save forensic prediction panels",
    )
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image resolution")
    return parser.parse_args()


def predict_single_image(model, image_path: Path, output_dir: Path, threshold: float, imgsz: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Deep Learning inference
    t0 = time.perf_counter()
    results = model.predict(source=str(image_path.resolve()), imgsz=imgsz, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    res = results[0]
    probs = res.probs.data.cpu().numpy()
    model_classes = model.names

    fake_idx = 1
    real_idx = 0
    for idx, name in model_classes.items():
        if "fake" in name.lower():
            fake_idx = idx
        elif "real" in name.lower():
            real_idx = idx

    fake_conf = float(probs[fake_idx])
    real_conf = float(probs[real_idx])

    is_fake = fake_conf >= threshold
    verdict = "FAKE / AI-GENERATED" if is_fake else "REAL / AUTHENTIC"
    verdict_conf = fake_conf if is_fake else real_conf

    # 2. Run Forensic Analysis
    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        print(f"Error: Could not read image: {image_path}")
        return

    ela_map_rgb, ela_score = compute_ela(image_path)
    fft_map_rgb, hf_ratio = compute_fft_spectrum(image_path)

    ela_map_bgr = cv2.cvtColor(ela_map_rgb, cv2.COLOR_RGB2BGR)
    fft_map_bgr = cv2.cvtColor(fft_map_rgb, cv2.COLOR_RGB2BGR)

    # 3. Create Multi-Panel Forensic Visualization
    panel_h, panel_w = 320, 320
    p1 = cv2.resize(orig_bgr, (panel_w, panel_h))
    p2 = cv2.resize(ela_map_bgr, (panel_w, panel_h))
    p3 = cv2.resize(fft_map_bgr, (panel_w, panel_h))

    # Add labels to panels
    cv2.putText(p1, "Input Image", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(p2, f"ELA (Score: {ela_score:.1f})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(p3, f"2D-FFT (HF: {hf_ratio:.1%})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    combined_panels = np.hstack([p1, p2, p3])

    # Header banner
    banner_h = 70
    banner = np.zeros((banner_h, combined_panels.shape[1], 3), dtype=np.uint8)
    banner_color = (0, 0, 200) if is_fake else (0, 160, 40)  # Red for fake, Green for real
    banner[:] = banner_color

    title_text = f"Verdict: {verdict} ({verdict_conf:.1%})"
    sub_text = f"File: {image_path.name} | Latency: {latency_ms:.1f}ms | Fake Prob: {fake_conf:.2%} | Real Prob: {real_conf:.2%}"

    cv2.putText(banner, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(banner, sub_text, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    full_output = np.vstack([banner, combined_panels])

    out_file = output_dir / f"pred_{image_path.stem}.jpg"
    cv2.imwrite(str(out_file), full_output)

    # Print summary to console
    status_icon = "🚨 [FAKE]" if is_fake else "✅ [REAL]"
    print(f"\n{status_icon} Image: {image_path.name}")
    print(f"  * Verdict:             {verdict}")
    print(f"  * Fake Probability:    {fake_conf:.2%}")
    print(f"  * Real Probability:    {real_conf:.2%}")
    print(f"  * ELA Discrepancy:     {ela_score:.2f}")
    print(f"  * FFT High-Freq Energy:{hf_ratio:.2%}")
    print(f"  * Inference Speed:     {latency_ms:.1f} ms")
    print(f"  * Forensic Proof Saved:{out_file.resolve()}")

    return verdict, fake_conf, out_file


def main():
    args = parse_arguments()

    img_path = Path(args.image)
    model_path = Path(args.model)
    output_dir = Path(args.output)

    if not model_path.is_file():
        last_path = model_path.parent / "last.pt"
        if last_path.is_file():
            print(f"Notice: '{model_path}' not found, falling back to '{last_path}'")
            model_path = last_path
        else:
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'.\n"
                "Please run 'src/train.py' first to train the detector."
            )

    if not img_path.exists():
        raise FileNotFoundError(f"Input image path does not exist: {img_path}")

    print(f"Loading Deepfake Detection Model: {model_path}...")
    model = YOLO(str(model_path.resolve()))

    if img_path.is_file():
        predict_single_image(model, img_path, output_dir, args.threshold, args.imgsz)
    elif img_path.is_dir():
        image_files = [p for p in img_path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
        if not image_files:
            print(f"No image files found in {img_path}")
            return
        print(f"Processing {len(image_files)} images from: {img_path}")
        for p in image_files:
            predict_single_image(model, p, output_dir, args.threshold, args.imgsz)


if __name__ == "__main__":
    main()
