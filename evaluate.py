
import os
import sys
import time
import argparse

import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.join(
    SCRIPT_DIR,
    "src"
)

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "model",
    "swinir_best.pth"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# IMPORT SWINIR
# ============================================================

from network_swinir import SwinIR


# ============================================================
# MODEL CONFIGURATION
# This MUST match the trained checkpoint.
# ============================================================

MODEL_CONFIG = {
    "img_size": 64,
    "patch_size": 1,
    "in_chans": 1,

    "embed_dim": 60,

    "depths": [6, 6, 6, 6],

    "num_heads": [6, 6, 6, 6],

    "window_size": 8,

    "mlp_ratio": 2,

    "qkv_bias": True,

    "qk_scale": None,

    "drop_rate": 0.0,

    "attn_drop_rate": 0.0,

    "drop_path_rate": 0.1,

    "ape": False,

    "patch_norm": True,

    "use_checkpoint": False,

    "upscale": 2,

    "img_range": 1.0,

    "upsampler": "pixelshuffledirect",

    "resi_connection": "1conv",
}


# ============================================================
# LOAD MODEL
# ============================================================

def create_model():

    model = SwinIR(
        **MODEL_CONFIG
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    # --------------------------------------------------------
    # Handle different checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "params" in checkpoint:

            state_dict = checkpoint[
                "params"
            ]

        elif "params_ema" in checkpoint:

            state_dict = checkpoint[
                "params_ema"
            ]

        else:

            # Check whether the dictionary itself
            # is a state dictionary.

            if all(
                isinstance(v, torch.Tensor)
                for v in checkpoint.values()
            ):

                state_dict = checkpoint

            else:

                raise RuntimeError(
                    "Could not identify model weights "
                    "inside checkpoint."
                )

    else:

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # --------------------------------------------------------
    # Remove possible DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    result = model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    print(
        "Missing keys   :",
        len(result.missing_keys)
    )

    print(
        "Unexpected keys:",
        len(result.unexpected_keys)
    )

    model.eval()

    return model


# ============================================================
# RESTORE SINGLE IMAGE
# ============================================================

@torch.inference_mode()
def restore_image(
    model,
    image
):

    # --------------------------------------------------------
    # Convert NumPy → Tensor
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        image.astype(np.float32)
    )

    if tensor.ndim == 2:

        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 3:

        raise ValueError(
            f"Expected 2D image, got shape {image.shape}"
        )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        DEVICE,
        non_blocking=True
    )

    # --------------------------------------------------------
    # SwinIR inference
    # --------------------------------------------------------

    output = model(
        tensor
    )

    # --------------------------------------------------------
    # Tensor → NumPy
    # --------------------------------------------------------

    output = (
        output
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Ensure valid image range
    # --------------------------------------------------------

    output = np.clip(
        output,
        0.0,
        1.0
    )

    output = output.astype(
        np.float32
    )

    return output


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "KLA SwinIR Image Restoration Evaluation"
        )
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing degraded .npy images"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for restored .npy images"
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(
        args.input_dir
    )

    output_dir = os.path.abspath(
        args.output_dir
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not os.path.isdir(input_dir):

        raise FileNotFoundError(
            f"Input directory does not exist:\n{input_dir}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find input images
    # --------------------------------------------------------

    files = sorted([
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith(".npy")
    ])

    if len(files) == 0:

        raise RuntimeError(
            "No .npy images found in input directory."
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print("=" * 70)
    print("KLA SWINIR EVALUATION")
    print("=" * 70)

    print()
    print("Input directory:")
    print(input_dir)

    print()
    print("Output directory:")
    print(output_dir)

    print()
    print("Input images :", len(files))

    print()
    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = torch.device(
            "cpu"
        )

        print(
            "⚠️ GPU unavailable — using CPU"
        )

    global DEVICE

    DEVICE = device

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading SwinIR...")

    model = create_model()

    model = model.to(
        DEVICE
    )

    print(
        "Model device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    successful = 0
    failed = 0

    total_time = 0.0

    print()
    print("-" * 70)
    print("STARTING INFERENCE")
    print("-" * 70)

    for index, filename in enumerate(
        files,
        start=1
    ):

        input_path = os.path.join(
            input_dir,
            filename
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        try:

            image = np.load(
                input_path
            )

            start_time = time.perf_counter()

            restored = restore_image(
                model,
                image
            )

            # Synchronize GPU before measuring
            if DEVICE.type == "cuda":

                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start_time
            )

            np.save(
                output_path,
                restored
            )

            total_time += elapsed

            successful += 1

            if (
                index <= 5
                or index % 25 == 0
                or index == len(files)
            ):

                print(
                    f"[{index:03d}/{len(files):03d}] "
                    f"{filename} | "
                    f"{image.shape} → "
                    f"{restored.shape} | "
                    f"{elapsed:.4f} sec"
                )

        except Exception as exc:

            failed += 1

            print(
                f"[ERROR] {filename}: {exc}"
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    if successful > 0:

        average_time = (
            total_time /
            successful
        )

        throughput = (
            successful /
            total_time
        )

    else:

        average_time = 0.0
        throughput = 0.0

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("KLA SWINIR EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Total input images :",
        len(files)
    )

    print(
        "Successful         :",
        successful
    )

    print(
        "Failed             :",
        failed
    )

    print(
        "Total inference    :",
        f"{total_time:.4f} sec"
    )

    print(
        "Average/image      :",
        f"{average_time:.4f} sec"
    )

    print(
        "Throughput         :",
        f"{throughput:.2f} images/sec"
    )

    print()
    print(
        "Output directory:"
    )

    print(
        output_dir
    )

    print()

    if failed == 0:

        print(
            "🏆 ALL IMAGES RESTORED SUCCESSFULLY"
        )

    else:

        print(
            "⚠️ SOME IMAGES FAILED"
        )

    print("=" * 70)


if __name__ == "__main__":

    main()
