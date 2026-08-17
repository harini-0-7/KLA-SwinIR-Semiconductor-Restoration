# KLA SwinIR — Verified Experimental Results

## Semiconductor Image Restoration

This document summarizes the verified experimental results obtained
from the KLA SwinIR semiconductor image restoration pipeline.

---

## 1. Dataset

| Category | Quantity |
|---|---:|
| Training pairs | 3,200 |
| Training samples | 2,880 |
| Validation samples | 320 |
| Test images | 400 |
| Input resolution | 128 × 128 |
| Ground-truth resolution | 256 × 256 |
| Image type | Grayscale |
| File format | NumPy `.npy` |

---

## 2. Model Configuration

| Parameter | Configuration |
|---|---|
| Architecture | SwinIR |
| Input channels | 1 |
| Embedding dimension | 60 |
| Depths | `[6, 6, 6, 6]` |
| Attention heads | `[6, 6, 6, 6]` |
| Upsampler | PixelShuffleDirect |
| Upscaling factor | ×2 |
| Framework | PyTorch |

---

## 3. Training

The trained checkpoint was successfully loaded with zero missing keys
and zero unexpected keys.

| Parameter | Result |
|---|---:|
| Best epoch | 13 |
| Training loss | 0.0371406819 |
| Validation loss | 0.0343663945 |
| Checkpoint size | 23.49 MB |

Checkpoint:

`model/swinir_best.pth`

---

## 4. Validation Results

The complete validation set contained 320 images.

| Metric | Average | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|
| PSNR | **29.1822 dB** | 28.8226 dB | 11.4916 dB | 46.8122 dB |
| SSIM | **0.764353** | 0.810351 | — | — |
| LPIPS | **0.299637** | 0.242198 | 0.015323 | 1.012695 |

### Metric interpretation

- **PSNR:** Higher is better.
- **SSIM:** Higher is better.
- **LPIPS:** Lower is better.

These metrics jointly evaluate pixel-level reconstruction,
structural similarity, and perceptual similarity.

---

## 5. Sample Evaluation

Sample:

`001692.npy`

| Metric | Result |
|---|---:|
| PSNR | **29.5570 dB** |
| SSIM | **0.904259** |
| Inference time | 0.2085 sec |

The sample comparison demonstrates the transformation:

**Degraded 128 × 128 → SwinIR Restored 256 × 256 → Ground Truth 256 × 256**

---

## 6. Test Set Inference

The trained model was evaluated on the complete test set.

| Result | Value |
|---|---:|
| Test images | 400 |
| Successful | **400** |
| Failed | **0** |
| Success rate | **100%** |
| Output resolution | 256 × 256 |
| Output data type | float32 |
| Output format | `.npy` |

### Output verification

All 400 generated files were independently verified.

- Correct number of files: **400**
- Correct output shape `(256, 256)`: **400**
- Invalid shapes: **0**
- Invalid values: **0**

---

## 7. Inference Performance

Hardware used for the verified GPU evaluation:

**NVIDIA Tesla T4**

| Performance metric | Result |
|---|---:|
| Total test images | 400 |
| Total GPU inference time | 88.3738 sec |
| Average/image | 0.2209 sec |
| Throughput | 4.53 images/sec |

---

## 8. Standalone Evaluation

The repository includes `evaluate.py` for independent inference.

A standalone test was performed using 10 input images.

| Result | Value |
|---|---:|
| Input images | 10 |
| Successful | **10** |
| Failed | **0** |
| Average inference | 0.2297 sec/image |
| Throughput | 4.35 images/sec |
| GPU | NVIDIA Tesla T4 |

The standalone evaluation script successfully loaded the trained
checkpoint and restored all test images.

---

## 9. Final Verified Summary

| Metric | Verified Result |
|---|---:|
| Training pairs | 3,200 |
| Validation images | 320 |
| Test images | 400 |
| Best epoch | 13 |
| Average PSNR | **29.1822 dB** |
| Average SSIM | **0.764353** |
| Average LPIPS | **0.299637** |
| Test restoration | **400 / 400** |
| Test success rate | **100%** |
| Output resolution | **256 × 256** |
| Average inference | **0.2209 sec/image** |
| Throughput | **4.53 images/sec** |
| GPU | **NVIDIA Tesla T4** |

---

## 10. Reproducibility

The repository provides:

- SwinIR architecture
- Training script
- Evaluation script
- Trained checkpoint
- Dependency specification
- Inference instructions
- Verified experimental results

The evaluation pipeline accepts degraded `.npy` images and produces
restored 256 × 256 `.npy` outputs.

---

## 11. Project Completion Status

- [x] Dataset preparation
- [x] Training
- [x] SwinIR checkpoint creation
- [x] Checkpoint architecture verification
- [x] PSNR evaluation
- [x] SSIM evaluation
- [x] LPIPS evaluation
- [x] Full 400-image test inference
- [x] Output verification
- [x] Standalone evaluator verification
- [x] GitHub repository
- [x] README documentation
- [x] Verified experimental results

---

## 12. Conclusion

The KLA SwinIR pipeline successfully restores degraded grayscale
semiconductor inspection images from 128 × 128 inputs to 256 × 256
outputs.

The complete test set achieved 400/400 successful restorations, while
the validation evaluation produced an average PSNR of 29.1822 dB,
SSIM of 0.764353, and LPIPS of 0.299637.

The results demonstrate a functional end-to-end deep-learning
restoration pipeline suitable for further optimization and
semiconductor inspection research.

