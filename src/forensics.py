#!/usr/bin/env python3
"""
src/forensics.py
-----------------
Explainable AI (XAI) & Image Forensics Analysis Module for Deepfake & AI Detection.

Implements three core forensic methods:
1. Error Level Analysis (ELA): Identifies digital manipulation by measuring JPEG compression error rate.
2. 2D Fast Fourier Transform (FFT) Magnitude Spectrum: Exposes periodic grid/checkerboard artifacts from GAN/Diffusion decoders.
3. High-Pass Noise Residual Analysis: Detects lack of natural camera sensor noise (PRNU) and synthetic skin smoothing.
"""

import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def compute_ela(image_input, quality: int = 90, scale: int = 15) -> tuple[np.ndarray, float]:
    """
    Compute Error Level Analysis (ELA).
    Compares the original image against a resaved JPEG version.
    Manipulated/AI areas show significantly different error rates compared to natural camera capture.
    Returns:
        ela_rgb (np.ndarray): Visual ELA heat map (RGB, uint8)
        ela_score (float): Mean discrepancy metric (higher implies suspicious variance)
    """
    if isinstance(image_input, (str, Path)):
        orig_pil = Image.open(str(image_input)).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        orig_pil = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.shape[2] == 3 else image_input)
    else:
        orig_pil = image_input.convert("RGB")

    # Resave to in-memory buffer at specified JPEG quality
    buffer = io.BytesIO()
    orig_pil.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved_pil = Image.open(buffer)

    # Compute absolute pixel difference
    diff = ImageChops.difference(orig_pil, resaved_pil)

    # Enhance difference to make artifacts human-visible
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    enhancer = ImageEnhance.Brightness(diff)
    ela_pil = enhancer.enhance(scale)

    ela_np = np.array(ela_pil)
    ela_score = float(np.mean(np.array(diff)))

    return ela_np, ela_score


def compute_fft_spectrum(image_input) -> tuple[np.ndarray, float]:
    """
    Compute 2D Fast Fourier Transform (FFT) Power Spectrum.
    Exposes periodic checkerboard / upsampling grid artifacts typical of GANs (StyleGAN)
    and latent diffusion models that are imperceptible in spatial domain.
    Returns:
        magnitude_spectrum_vis (np.ndarray): Visualized 2D power spectrum (RGB, uint8)
        hf_energy_ratio (float): Ratio of high-frequency energy to total energy
    """
    if isinstance(image_input, (str, Path)):
        img_bgr = cv2.imread(str(image_input))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            gray = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_input.copy()
    else:
        gray = np.array(image_input.convert("L"))

    # Compute 2D Fast Fourier Transform
    dft = np.fft.fft2(gray.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)

    # Magnitude spectrum
    mag = np.abs(dft_shift)
    mag_log = np.log(mag + 1.0)

    # Normalize to 0-255 for visualization with Jet colormap
    norm_mag = cv2.normalize(mag_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    spectrum_colored = cv2.applyColorMap(norm_mag, cv2.COLORMAP_JET)
    spectrum_rgb = cv2.cvtColor(spectrum_colored, cv2.COLOR_BGR2RGB)

    # Calculate high-frequency energy ratio
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    low_freq_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= r**2

    total_energy = np.sum(mag**2) + 1e-8
    low_energy = np.sum((mag * low_freq_mask) ** 2)
    hf_energy = total_energy - low_energy
    hf_ratio = float(hf_energy / total_energy)

    return spectrum_rgb, hf_ratio


def compute_noise_residual(image_input) -> tuple[np.ndarray, float]:
    """
    Compute High-Pass Noise Residual (PRNU proxy).
    Extracts sensor noise pattern using a high-pass Gaussian difference filter.
    Authentic photographs possess organic sensor noise, whereas AI synthesis has synthetic smoothness.
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        img = np.array(image_input.convert("RGB"))

    # Gaussian blur as low-pass filter
    blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
    # High pass = Original - Low Pass
    residual = cv2.absdiff(img, blurred)

    # Normalize residual for clear visibility
    res_norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    noise_variance = float(np.var(residual))

    return res_norm, noise_variance


def generate_forensic_suite(image_input) -> dict:
    """
    Run complete forensic test suite on an image and return visual maps and metrics.
    """
    ela_map, ela_score = compute_ela(image_input)
    fft_map, hf_ratio = compute_fft_spectrum(image_input)
    noise_map, noise_var = compute_noise_residual(image_input)

    return {
        "ela_map": ela_map,
        "ela_score": ela_score,
        "fft_map": fft_map,
        "fft_hf_ratio": hf_ratio,
        "noise_map": noise_map,
        "noise_variance": noise_var,
    }

