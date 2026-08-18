
import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

NETWORK_FILE = os.path.join(MODEL_DIR, "network_swinir.py")
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "swinir_best.pth")

sys.path.insert(0, MODEL_DIR)

from network_swinir import SwinIR


# ============================================================
# EXACT TRAINED MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "upscale": 2,
    "in_chans": 1,
    "img_size": 64,
    "window_size": 8,
    "img_range": 1.0,
    "depths": [6, 6, 6, 6],
    "embed_dim": 60,
    "num_heads": [6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": "pixelshuffledirect",
    "resi_connection": "1conv"
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):

    if not os.path.isfile(NETWORK_FILE):
        raise FileNotFoundError(
            "Missing network_swinir.py: " + NETWORK_FILE
        )

    if not os.path.isfile(CHECKPOINT_FILE):
        raise FileNotFoundError(
            "Missing swinir_best.pth: " + CHECKPOINT_FILE
        )

    model = SwinIR(**MODEL_CONFIG)

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "params" in checkpoint:
            state_dict = checkpoint["params"]

        elif "params_ema" in checkpoint:
            state_dict = checkpoint["params_ema"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing or unexpected:

        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            "Missing keys: " + str(missing) + "\n"
            "Unexpected keys: " + str(unexpected)
        )

    model.eval()
    model.to(device)

    return model


# ============================================================
# RESTORE ONE NPY IMAGE
# ============================================================

def restore_image(model, array, device):

    array = np.asarray(
        array,
        dtype=np.float32
    )

    if array.ndim == 3:

        if array.shape[-1] == 1:
            array = array[:, :, 0]

        else:
            raise ValueError(
                "Input must be grayscale with shape (H,W) "
                "or (H,W,1)."
            )

    if array.ndim != 2:

        raise ValueError(
            "Input must have shape (H,W) or (H,W,1)."
        )

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    array = np.clip(
        array,
        0.0,
        1.0
    )

    tensor = torch.from_numpy(
        array
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(
        device=device,
        dtype=torch.float32
    )

    with torch.no_grad():

        output = model(tensor)

    output = output.squeeze()

    output = output.detach().cpu().numpy()

    output = np.nan_to_num(
        output,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

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
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 3:

        print(
            "Usage: python run.py <input-dir> <output-dir>"
        )

        sys.exit(1)

    input_dir = os.path.abspath(
        sys.argv[1]
    )

    output_dir = os.path.abspath(
        sys.argv[2]
    )

    if not os.path.isdir(input_dir):

        raise FileNotFoundError(
            "Input directory does not exist: "
            + input_dir
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Input :", input_dir)
    print("Output:", output_dir)

    model = load_model(device)

    input_files = sorted(
        filename
        for filename in os.listdir(input_dir)
        if filename.lower().endswith(".npy")
    )

    if not input_files:

        raise RuntimeError(
            "No .npy files found in input directory."
        )

    print(
        "Input files:",
        len(input_files)
    )

    successful = 0
    failed = 0

    for filename in input_files:

        input_path = os.path.join(
            input_dir,
            filename
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        try:

            array = np.load(
                input_path
            )

            restored = restore_image(
                model,
                array,
                device
            )

            if restored.ndim != 2:

                raise RuntimeError(
                    "Restored output must be 2D. "
                    "Got: "
                    + str(restored.shape)
                )

            if not np.isfinite(
                restored
            ).all():

                raise RuntimeError(
                    "Output contains NaN or Inf."
                )

            if (
                restored.min() < 0.0
                or restored.max() > 1.0
            ):

                raise RuntimeError(
                    "Output values outside [0,1]."
                )

            np.save(
                output_path,
                restored
            )

            successful += 1

            print(
                f"[{successful}/{len(input_files)}] "
                f"{filename}"
            )

        except Exception as exc:

            failed += 1

            print(
                "FAILED:",
                filename,
                "|",
                repr(exc)
            )

    print("=" * 70)
    print("RESTORATION COMPLETE")
    print("=" * 70)

    print(
        "Total   :",
        len(input_files)
    )

    print(
        "Success :",
        successful
    )

    print(
        "Failed  :",
        failed
    )

    if failed != 0:

        sys.exit(1)


if __name__ == "__main__":
    main()
