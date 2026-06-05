"""
train.py — Training entry point for DeepPermNet-ViT

Usage:
    python src/train.py [--epochs N] [--batch-size N] [--lr FLOAT]

All defaults fall back to .env values.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dotenv import load_dotenv

from dataset import DeepPermNetDataModule
from model import DeepPermNetViT

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepPermNet-ViT")
    parser.add_argument("--epochs",     type=int,   default=int(os.getenv("MAX_EPOCHS",    15)))
    parser.add_argument("--batch-size", type=int,   default=int(os.getenv("BATCH_SIZE",    32)))
    parser.add_argument("--lr",         type=float, default=float(os.getenv("LEARNING_RATE", 1e-4)))
    parser.add_argument("--d-model",    type=int,   default=int(os.getenv("D_MODEL",   512)))
    parser.add_argument("--nhead",      type=int,   default=int(os.getenv("NHEAD",     8)))
    parser.add_argument("--num-layers", type=int,   default=int(os.getenv("NUM_LAYERS", 4)))
    parser.add_argument("--save-dir",   type=str,   default=os.getenv("MODEL_SAVE_DIR", "./models/ViT"))
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_module = DeepPermNetDataModule(batch_size=args.batch_size)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = DeepPermNetViT(
        learning_rate=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename="deeppermnet-vit-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stopping],
        default_root_dir=args.save_dir,
    )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    print(f"Starting DeepPermNet-ViT training for {args.epochs} epochs …")
    trainer.fit(model, datamodule=data_module)

    # ------------------------------------------------------------------
    # Save final weights & ONNX export
    # ------------------------------------------------------------------
    weights_path = os.path.join(args.save_dir, "deep_perm_net.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved weights → {weights_path}")

    onnx_path = os.path.join(args.save_dir, "deep_perm_net.onnx")
    dummy_input = torch.randn(1, 9, 3, 64, 64)
    torch.onnx.export(model.cpu(), dummy_input, onnx_path)
    print(f"Exported ONNX  → {onnx_path}")


if __name__ == "__main__":
    main()
