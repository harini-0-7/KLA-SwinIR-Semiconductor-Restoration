#!/usr/bin/env python3

"""
KLA SwinIR Training Script
--------------------------

Training pipeline for grayscale semiconductor image restoration.

Input:
    128x128 degraded .npy images

Target:
    256x256 ground-truth .npy images

Dataset:
    KLA_PROJECT/train/train/NoisyLR
    KLA_PROJECT/train/train/GT

Model:
    SwinIR
    2x PixelShuffleDirect upsampling

This script is designed to run as a standalone Python program.
"""

import os
import sys
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class KLADataset(Dataset):

    def __init__(self, gt_dir, lr_dir):

        self.gt_dir = gt_dir
        self.lr_dir = lr_dir

        gt_files = {
            f for f in os.listdir(gt_dir)
            if f.endswith(".npy")
        }

        lr_files = {
            f for f in os.listdir(lr_dir)
            if f.endswith(".npy")
        }

        self.files = sorted(gt_files.intersection(lr_files))

        if len(self.files) == 0:
            raise RuntimeError(
                "No paired .npy files found.\n"
                f"GT directory : {gt_dir}\n"
                f"LR directory : {lr_dir}"
            )

        print("Paired samples:", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        lr_path = os.path.join(
            self.lr_dir,
            filename
        )

        gt_path = os.path.join(
            self.gt_dir,
            filename
        )

        lr = np.load(lr_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        lr = torch.from_numpy(lr).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return lr, gt


# ============================================================
# SWINIR IMPORT
# ============================================================

def load_swinir(repo_root):

    src_dir = os.path.join(
        repo_root,
        "src"
    )

    if not os.path.exists(src_dir):
        raise FileNotFoundError(
            f"SwinIR source directory not found: {src_dir}"
        )

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from network_swinir import SwinIR

    return SwinIR


# ============================================================
# MODEL
# ============================================================

def create_model(SwinIR):

    model = SwinIR(
        upscale=2,
        in_chans=1,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
        resi_connection="1conv"
    )

    return model


# ============================================================
# LOSS
# ============================================================

class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-6):

        super().__init__()

        self.eps = eps

    def forward(self, prediction, target):

        diff = prediction - target

        loss = torch.sqrt(
            diff * diff + self.eps
        )

        return loss.mean()


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():

        for lr, gt in loader:

            lr = lr.to(
                device,
                non_blocking=True
            )

            gt = gt.to(
                device,
                non_blocking=True
            )

            prediction = model(lr)

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0
            )

            loss = criterion(
                prediction,
                gt
            )

            total_loss += (
                loss.item() * lr.size(0)
            )

            count += lr.size(0)

    return total_loss / count


# ============================================================
# TRAINING
# ============================================================

def train(args):

    set_seed(args.seed)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("KLA SWINIR TRAINING")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "WARNING: CUDA unavailable. "
            "Training will use CPU."
        )

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    gt_dir = os.path.abspath(
        args.gt_dir
    )

    lr_dir = os.path.abspath(
        args.lr_dir
    )

    checkpoint_dir = os.path.abspath(
        args.checkpoint_dir
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    print("\nGT directory:")
    print(gt_dir)

    print("\nLR directory:")
    print(lr_dir)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = KLADataset(
        gt_dir,
        lr_dir
    )

    validation_size = int(
        len(dataset) * args.validation_split
    )

    training_size = (
        len(dataset) -
        validation_size
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(
            args.seed
        )
    )

    print("\nDataset:")
    print("Total:", len(dataset))
    print("Training:", len(train_dataset))
    print("Validation:", len(val_dataset))

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    SwinIR = load_swinir(
        args.repo_root
    )

    model = create_model(
        SwinIR
    )

    model = model.to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nModel parameters:")
    print(
        f"{parameters / 1e6:.2f} M"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = CharbonnierLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    best_val_loss = float("inf")

    best_checkpoint = os.path.join(
        checkpoint_dir,
        "swinir_best.pth"
    )

    for epoch in range(
        1,
        args.epochs + 1
    ):

        start_time = time.time()

        model.train()

        running_loss = 0.0

        for batch_idx, (lr, gt) in enumerate(
            train_loader,
            start=1
        ):

            lr = lr.to(
                device,
                non_blocking=True
            )

            gt = gt.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(lr)

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0
            )

            loss = criterion(
                prediction,
                gt
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:

                print(
                    f"Epoch {epoch:02d}/{args.epochs} "
                    f"| Batch {batch_idx:04d}/{len(train_loader)} "
                    f"| Loss {loss.item():.5f}"
                )

        train_loss = (
            running_loss /
            len(train_loader)
        )

        val_loss = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = (
            time.time() -
            start_time
        )

        print(
            f"\nEpoch {epoch:02d}/{args.epochs} "
            f"Train Loss : {train_loss:.6f} "
            f"Val Loss : {val_loss:.6f} "
            f"LR : {current_lr:.8f} "
            f"Time : {elapsed:.1f} sec"
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            torch.save(
                checkpoint,
                best_checkpoint
            )

            print(
                "🏆 New best model saved!"
            )

            print(
                "Best validation loss:",
                best_val_loss
            )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("Best validation loss:")
    print(best_val_loss)

    print("\nBest checkpoint:")
    print(best_checkpoint)


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train SwinIR for KLA image restoration"
    )

    parser.add_argument(
        "--repo_root",
        type=str,
        default=os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--lr_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./model"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5
    )

    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.10
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


if __name__ == "__main__":

    arguments = parse_args()

    train(arguments)
