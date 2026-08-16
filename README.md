# KLA SwinIR - Semiconductor Image Restoration

## AI-Based Restoration of Degraded Semiconductor Inspection Images

This project implements a SwinIR-based image restoration pipeline for
grayscale semiconductor inspection images.

The model takes a degraded low-resolution image and reconstructs a
higher-resolution restored image.

---

## 1. Problem

Semiconductor inspection images can suffer from noise and spatial
resolution reduction. These degradations can hide fine structures and
make defect inspection difficult.

The objective is to restore degraded images while preserving important
structural information.

---

## 2. Proposed Solution

The project uses SwinIR (Swin Transformer for Image Restoration).

Pipeline:

    Degraded Image
          |
          v
    128 x 128 Input
          |
          v
       SwinIR
          |
          v
    Feature Extraction
          |
          v
    Transformer Blocks
          |
          v
    PixelShuffle Upsampling
          |
          v
    256 x 256 Output
          |
          v
    Restored Image

The model operates on single-channel grayscale images.

---

## 3. Dataset

Training dataset:

    Ground Truth images : 3200
    NoisyLR images      : 3200

Paired samples:

    3200

Data split:

    Training   : 2880
    Validation : 320

Image dimensions:

    Ground Truth : 256 x 256
    Input        : 128 x 128

Test dataset:

    Test images : 400
    Input       : 128 x 128
    Output      : 256 x 256

Images are stored as NumPy .npy files.

---

## 4. SwinIR Configuration

Model configuration used for the trained checkpoint:

    Input channels : 1
    Embedding dim  : 60

    Depths         : [6, 6, 6, 6]
    Attention heads: [6, 6, 6, 6]

    Upsampler      : pixelshuffledirect
    Upscale factor : 2

The model contains approximately 0.90 million parameters.

Checkpoint size:

    23.49 MB

---

## 5. Training

Framework:

    PyTorch

GPU:

    NVIDIA Tesla T4

CUDA:

    12.8

PyTorch:

    2.11.0+cu128

Best checkpoint:

    Epoch          : 13
    Training loss  : 0.0371406819
    Validation loss: 0.0343663945

Checkpoint:

    checkpoints/swinir_best.pth

---

## 6. Evaluation Metrics

The project evaluates restoration quality using:

### PSNR

Peak Signal-to-Noise Ratio.

Higher values indicate better pixel-level reconstruction.

### SSIM

Structural Similarity Index.

Higher values indicate greater structural similarity.

### LPIPS

Learned Perceptual Image Patch Similarity.

Lower values indicate greater perceptual similarity.

---

## 7. Validation Results

Validation dataset:

    320 images

Average results:

    PSNR  : 29.1822 dB
    SSIM  : 0.764353
    LPIPS : 0.299637

PSNR statistics:

    Minimum : 11.4916 dB
    Maximum : 46.8122 dB
    Median  : 28.8226 dB

Median SSIM:

    0.810351

---

## 8. Sample Restoration

Sample:

    001692.npy

Sample results:

    PSNR : 29.5570 dB
    SSIM : 0.904259

The visual comparison contains:

    Degraded Input
          |
          v
    SwinIR Output
          |
          v
      Ground Truth

Comparison image:

    results/swinir_comparison_001692.png

---

## 9. Test Set Results

The final model was evaluated on all 400 test images.

    Total images : 400
    Successful   : 400
    Failed       : 0

Every test image was successfully restored.

Output:

    256 x 256
    float32
    NumPy .npy

Output values are clipped to the valid [0, 1] range.

---

## 10. Inference Performance

Hardware:

    NVIDIA Tesla T4

Full test set:

    Total GPU time : 88.3738 seconds
    Average/image  : 0.2209 seconds
    Throughput     : 4.53 images/second

Standalone evaluation test:

    Images tested : 10
    Successful    : 10
    Failed        : 0

    Average/image : 0.2297 seconds
    Throughput    : 4.35 images/second

---

## 11. Standalone Evaluation

The repository contains:

    evaluate.py

The evaluation script accepts:

    --input_dir
    --output_dir

Example:

    python evaluate.py --input_dir ./test_images --output_dir ./restored_outputs

The script:

1. Loads the trained SwinIR checkpoint.
2. Detects CUDA when available.
3. Reads .npy grayscale images.
4. Performs image restoration.
5. Upscales 128 x 128 inputs to 256 x 256.
6. Clips output values to [0, 1].
7. Saves restored images as float32 .npy.
8. Reports inference time and throughput.

The standalone evaluation script was successfully tested without manual
source-code modification.

---

## 12. Repository Structure

    KLA_SwinIR/
    |
    +-- README.md
    +-- evaluate.py
    +-- train.py
    +-- requirements.txt
    |
    +-- checkpoints/
    |   +-- swinir_best.pth
    |
    +-- model/
    |   +-- network_swinir.py
    |
    +-- results/
        +-- restored_test/
        +-- swinir_comparison_001692.png

---

## 13. Installation

Create a Python environment and install the dependencies:

    pip install -r requirements.txt

CUDA-enabled PyTorch is recommended when an NVIDIA GPU is available.

---

## 14. Running Inference

Place degraded .npy images in an input directory:

    test_images/
    +-- 000000.npy
    +-- 000001.npy
    +-- 000002.npy

Run:

    python evaluate.py --input_dir ./test_images --output_dir ./restored_outputs

The restored images will be generated inside:

    restored_outputs/

---

## 15. Input Format

Input files:

    Format    : NumPy .npy
    Channels  : 1
    Resolution: 128 x 128
    Data type : float32

The degraded input may contain values outside the [0, 1] range because
of image degradation and noise.

---

## 16. Output Format

Output files:

    Format    : NumPy .npy
    Channels  : 1
    Resolution: 256 x 256
    Data type : float32
    Range     : [0, 1]

---

## 17. Reproducibility

The repository is intended to provide:

- Training code
- Evaluation code
- Trained model checkpoint
- Requirements file
- Example restoration
- Restored test outputs
- Complete inference instructions

The standalone evaluation script can be executed independently for
model benchmarking.

---

## 18. Results Summary

| Metric | Result |
|---|---:|
| Validation images | 320 |
| Average PSNR | 29.1822 dB |
| Average SSIM | 0.764353 |
| Average LPIPS | 0.299637 |
| Test images | 400 |
| Successful restoration | 400/400 |
| Failed images | 0 |
| Output resolution | 256 x 256 |
| Average test inference | 0.2209 sec/image |
| Test throughput | 4.53 images/sec |
| GPU | NVIDIA Tesla T4 |

---

## 19. Project Status

    Dataset preparation     : COMPLETE
    SwinIR implementation   : COMPLETE
    Training                : COMPLETE
    Checkpoint validation   : COMPLETE
    PSNR evaluation         : COMPLETE
    SSIM evaluation         : COMPLETE
    LPIPS evaluation        : COMPLETE
    Test inference          : COMPLETE
    400/400 restoration     : COMPLETE
    Standalone evaluator    : VERIFIED
    GitHub packaging        : IN PROGRESS

---

## 20. Hackathon Submission

This repository was prepared for the KLA semiconductor image
restoration challenge.

The project focuses on efficient deep-learning-based restoration of
degraded microscopic semiconductor inspection images.

