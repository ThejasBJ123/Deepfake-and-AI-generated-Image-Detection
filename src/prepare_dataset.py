#!/usr/bin/env python3
"""
src/prepare_dataset.py
-----------------------
Dataset Preprocessing and Synthesis Pipeline for Deepfake and AI-Generated Image Detection.

Organizes images into standard Deep Learning Image Classification hierarchy:
  data/deepfake_dataset/
  ├── train/
  │   ├── real/
  │   └── fake/
  ├── val/
  │   ├── real/
  │   └── fake/
  └── test/
      ├── real/
      └── fake/

Features:
1. Ingests existing Real & Fake image directories (e.g. CIFAKE, FaceForensics++, 140k Real/Fake).
2. Automatically harvests real human face photographs from local WIDER FACE archive.
3. Synthesizes photorealistic Deepfake and AI-generated image artifacts:
   - Facial boundary seam blending and warping (FaceSwap simulation)
   - 2D-FFT GAN periodic checkerboard frequency anomalies (StyleGAN deconvolution artifact simulation)
   - Latent Diffusion neural texture smoothing and high-frequency noise residuals
   - Compression & chromatic mismatch
"""

import argparse
import math
import random
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_all_images(folder: Path) -> list[Path]:
    """Recursively collect all image files inside folder."""
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS and p.is_file()]


def apply_deepfake_artifacts(img_rgb: np.ndarray, artifact_type: int) -> np.ndarray:
    """
    Apply realistic deepfake / AI-generation artifacts to a real image:
      Type 0: GAN Periodic Frequency Artifacts (upsampling grid/checkerboard)
      Type 1: Face-Swap Boundary Seam & Warping (blending boundary discontinuity)
      Type 2: Diffusion Neural Smoothing & Edge Sharpening (over-smoothed skin with artificial edges)
      Type 3: Chromatic & Color Spectrum Inconsistency
      Type 4: Multi-artifact composite (Face swap + GAN residual)
    """
    h, w, c = img_rgb.shape
    out = img_rgb.copy().astype(np.float32)

    if artifact_type == 0:
        # GAN Checkerboard Artifact: Periodic high-frequency grid across spatial channels
        freq = random.choice([4, 8, 16])
        y_grid, x_grid = np.ogrid[:h, :w]
        grid_pattern = np.sin(x_grid * (2 * math.pi / freq)) * np.cos(y_grid * (2 * math.pi / freq))
        strength = random.uniform(8.0, 18.0)
        out += np.repeat(grid_pattern[:, :, np.newaxis] * strength, 3, axis=2)

    elif artifact_type == 1:
        # Face-Swap Boundary Blending: Extract central facial region, warp, and blend with seam artifact
        cx, cy = w // 2, h // 2
        rx, ry = int(w * random.uniform(0.22, 0.35)), int(h * random.uniform(0.25, 0.38))
        x1, y1 = max(0, cx - rx), max(0, cy - ry)
        x2, y2 = min(w, cx + rx), min(h, cy + ry)

        face_roi = out[y1:y2, x1:x2].copy()
        # Slight color shift in face swap (lighting mismatch)
        shift = np.array([random.uniform(-15, 15), random.uniform(-10, 10), random.uniform(-15, 15)])
        face_roi = np.clip(face_roi + shift, 0, 255)

        # Bilateral filter to simulate synthetic deepfake skin
        face_uint8 = np.clip(face_roi, 0, 255).astype(np.uint8)
        smooth_face = cv2.bilateralFilter(face_uint8, d=9, sigmaColor=75, sigmaSpace=75)

        # Elliptical blending mask with edge artifact
        mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
        cv2.ellipse(
            mask,
            ((x2 - x1) // 2, (y2 - y1) // 2),
            ((x2 - x1) // 2 - 2, (y2 - y1) // 2 - 2),
            0,
            0,
            360,
            1.0,
            -1,
        )
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        out[y1:y2, x1:x2] = out[y1:y2, x1:x2] * (1.0 - mask_3d) + smooth_face.astype(np.float32) * mask_3d

    elif artifact_type == 2:
        # Diffusion Neural Texture: Over-smoothed skin texture combined with high-contrast edge residuals
        uint8_img = np.clip(out, 0, 255).astype(np.uint8)
        smooth = cv2.bilateralFilter(uint8_img, d=11, sigmaColor=100, sigmaSpace=100)
        # Unsharp mask for artificial hyper-sharpening
        gaussian = cv2.GaussianBlur(smooth, (0, 0), 2.0)
        sharpened = cv2.addWeighted(smooth, 1.4, gaussian, -0.4, 0)
        out = sharpened.astype(np.float32)

    elif artifact_type == 3:
        # Chromatic & Color Spectrum Inconsistency (GAN color channel mismatch)
        out[:, :, 0] = np.clip(out[:, :, 0] * random.uniform(0.92, 1.08) + random.uniform(-8, 8), 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] * random.uniform(0.90, 1.10) + random.uniform(-10, 10), 0, 255)
        # Add high-frequency noise in blue channel (common in StyleGAN)
        noise = np.random.normal(0, random.uniform(3, 8), (h, w))
        out[:, :, 0] = np.clip(out[:, :, 0] + noise, 0, 255)

    else:
        # Composite artifact
        out = apply_deepfake_artifacts(out, 0)
        out = apply_deepfake_artifacts(out, 1)

    return np.clip(out, 0, 255).astype(np.uint8)


def generate_synthetic_benchmark_dataset(output_dir: Path, total_samples: int = 400):
    """
    Generate synthetic benchmark dataset containing realistic Real vs Deepfake/AI image pairs
    for offline verification and fast local training.
    """
    print(f"\n[Synthesis Mode] Generating {total_samples} balanced Real & Deepfake/AI image pairs...")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": int(total_samples * 0.70),
        "val": int(total_samples * 0.15),
        "test": total_samples - int(total_samples * 0.70) - int(total_samples * 0.15),
    }

    # Initialize directory structure
    for split in splits:
        (output_dir / split / "real").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "fake").mkdir(parents=True, exist_ok=True)

    skin_tones = [
        (255, 224, 189),
        (241, 194, 125),
        (224, 172, 105),
        (198, 134, 66),
        (141, 85, 36),
        (90, 56, 37),
    ]
    hair_colors = [(30, 20, 10), (70, 50, 30), (160, 120, 70), (200, 170, 110), (100, 30, 20)]

    idx = 0
    for split, count in splits.items():
        half = count // 2
        for i in range(half):
            # 1. Base photographic face creation
            w, h = 256, 256
            bg = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            pil_img = Image.new("RGB", (w, h), bg)
            draw = ImageDraw.Draw(pil_img)

            skin = random.choice(skin_tones)
            hair = random.choice(hair_colors)

            # Hair
            draw.chord([20, 10, w - 20, h - 20], 180, 360, fill=hair)
            # Face oval
            draw.ellipse([40, 40, w - 40, h - 30], fill=skin, outline=(40, 30, 30), width=2)
            # Eyes
            draw.ellipse([70, 100, 95, 115], fill=(255, 255, 255), outline=(30, 30, 30))
            draw.ellipse([80, 103, 90, 113], fill=(40, 30, 20))
            draw.ellipse([w - 95, 100, w - 70, 115], fill=(255, 255, 255), outline=(30, 30, 30))
            draw.ellipse([w - 90, 103, w - 80, 113], fill=(40, 30, 20))
            # Nose
            draw.line([(w // 2, 115), (w // 2, 145)], fill=(130, 80, 60), width=2)
            # Mouth
            draw.arc([80, 160, w - 80, 185], start=10, end=170, fill=(180, 40, 40), width=3)

            # Add subtle camera noise & compression to REAL image
            np_real = np.array(pil_img)
            cam_noise = np.random.normal(0, 3.5, np_real.shape).astype(np.float32)
            np_real = np.clip(np_real.astype(np.float32) + cam_noise, 0, 255).astype(np.uint8)

            # Save Real image
            real_path = output_dir / split / "real" / f"real_{split}_{i:04d}.jpg"
            cv2.imwrite(str(real_path), cv2.cvtColor(np_real, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # 2. Synthesize Deepfake/AI counterpart with forensic artifacts
            artifact_kind = random.randint(0, 4)
            np_fake = apply_deepfake_artifacts(np_real, artifact_kind)

            fake_path = output_dir / split / "fake" / f"fake_{split}_{i:04d}.jpg"
            cv2.imwrite(str(fake_path), cv2.cvtColor(np_fake, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            idx += 2

    print(f"Successfully generated {total_samples} samples across train/val/test splits in: {output_dir}")


def build_from_real_and_fake_sources(
    real_images: list[Path],
    fake_images: list[Path],
    output_dir: Path,
    max_per_class: int = 1000,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """Partition real and fake image collections into balanced train/val/test splits."""
    random.seed(seed)
    random.shuffle(real_images)
    random.shuffle(fake_images)

    if max_per_class and max_per_class < len(real_images):
        real_images = real_images[:max_per_class]
    if max_per_class and max_per_class < len(fake_images):
        fake_images = fake_images[:max_per_class]

    # Equalize count for perfect class balance
    common_count = min(len(real_images), len(fake_images))
    real_images = real_images[:common_count]
    fake_images = fake_images[:common_count]

    n_train = int(common_count * train_ratio)
    n_val = int(common_count * val_ratio)

    splits = {
        "train": (real_images[:n_train], fake_images[:n_train]),
        "val": (real_images[n_train : n_train + n_val], fake_images[n_train : n_train + n_val]),
        "test": (real_images[n_train + n_val :], fake_images[n_train + n_val :]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for split, (reals, fakes) in splits.items():
        real_dir = output_dir / split / "real"
        fake_dir = output_dir / split / "fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        for p in reals:
            shutil.copy2(p, real_dir / p.name)
        for p in fakes:
            shutil.copy2(p, fake_dir / p.name)

        stats[split] = (len(reals), len(fakes))

    print("\nDataset Partitioning Complete:")
    print("-" * 50)
    for s, (r_cnt, f_cnt) in stats.items():
        print(f"  {s:<6} split -> Real: {r_cnt:<5} | Fake: {f_cnt:<5} (Total: {r_cnt + f_cnt})")
    print("-" * 50)
    print(f"Dataset root: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Deepfake & AI Generated Image Dataset.")
    parser.add_argument("--real-dir", type=str, default="", help="Path to real images directory")
    parser.add_argument("--fake-dir", type=str, default="", help="Path to fake / AI-generated images directory")
    parser.add_argument("--output", type=str, default="data/deepfake_dataset", help="Output dataset directory")
    parser.add_argument("--max-images", type=int, default=1000, help="Max images per class")
    parser.add_argument("--generate-sample", action="store_true", help="Generate synthetic Real/Fake benchmark dataset")
    parser.add_argument("--sample-count", type=int, default=400, help="Number of synthetic samples to generate")
    parser.add_argument(
        "--synthesize-from-wider",
        action="store_true",
        help="Harvest real photos from data/raw/WIDER_train/images and generate deepfake pairs",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)

    # Mode 1: Generate synthetic Real vs Deepfake dataset for instant training
    if args.generate_sample:
        generate_synthetic_benchmark_dataset(out_dir, total_samples=args.sample_count)
        return

    # Mode 2: Synthesize fakes directly from local WIDER FACE archive
    wider_images_path = Path("data/raw/WIDER_train/images")
    if args.synthesize_from_wider or (not args.real_dir and wider_images_path.is_dir()):
        print(f"\n[Auto Mode] Sourcing real images from WIDER FACE: {wider_images_path}")
        all_real_photos = find_all_images(wider_images_path)
        if all_real_photos:
            random.seed(42)
            random.shuffle(all_real_photos)
            selected_real = all_real_photos[: args.max_images]

            temp_fakes_dir = Path("data/raw/synthetic_fakes")
            temp_fakes_dir.mkdir(parents=True, exist_ok=True)
            print(f"Generating {len(selected_real)} Deepfake / AI-manipulated counterparts...")

            fake_list = []
            for idx, r_path in enumerate(selected_real):
                bgr = cv2.imread(str(r_path))
                if bgr is None:
                    continue
                # Resize if excessively large for fast, high-quality training
                h, w = bgr.shape[:2]
                if max(h, w) > 512:
                    scale = 512.0 / max(h, w)
                    bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                artifact_kind = random.randint(0, 4)
                fake_rgb = apply_deepfake_artifacts(rgb, artifact_kind)

                f_name = f"fake_{r_path.stem}.jpg"
                fake_save_path = temp_fakes_dir / f_name
                cv2.imwrite(
                    str(fake_save_path),
                    cv2.cvtColor(fake_rgb, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(85, 95)],
                )
                fake_list.append(fake_save_path)

            print(f"Generated {len(fake_list)} deepfake images.")
            build_from_real_and_fake_sources(
                real_images=selected_real,
                fake_images=fake_list,
                output_dir=out_dir,
                max_per_class=args.max_images,
            )
            return

    # Mode 3: Ingest user-provided real and fake image folders
    real_folder = Path(args.real_dir) if args.real_dir else Path("data/raw/real")
    fake_folder = Path(args.fake_dir) if args.fake_dir else Path("data/raw/fake")

    real_imgs = find_all_images(real_folder)
    fake_imgs = find_all_images(fake_folder)

    if not real_imgs or not fake_imgs:
        print(f"Notice: Image sources not found.")
        print(f"  Real images found: {len(real_imgs)} in {real_folder}")
        print(f"  Fake images found: {len(fake_imgs)} in {fake_folder}")
        print("\nTip: Run with `--generate-sample` to generate a self-contained local benchmark dataset:")
        print("  python src/prepare_dataset.py --generate-sample --sample-count 400")
        sys.exit(1)

    build_from_real_and_fake_sources(
        real_images=real_imgs,
        fake_images=fake_imgs,
        output_dir=out_dir,
        max_per_class=args.max_images,
    )


if __name__ == "__main__":
    main()
