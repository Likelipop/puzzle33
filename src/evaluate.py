"""
evaluate.py — Patch-level accuracy evaluation for DeepPermNet-ViT

Usage:
    python src/evaluate.py --checkpoint ./models/ViT/deep_perm_net.pth
    python src/evaluate.py --checkpoint ./models/ViT/last.ckpt  (Lightning checkpoint)
    python src/evaluate.py  # uses best checkpoint discovered automatically
"""

import os
import argparse
import torch
from dotenv import load_dotenv

from dataset import DeepPermNetDataModule
from model import DeepPermNetViT

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepPermNet-ViT accuracy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pth (state-dict) or .ckpt (Lightning) checkpoint file.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.getenv("MODEL_SAVE_DIR", "./models/ViT"),
        help="Directory to search for the latest checkpoint if --checkpoint is not given.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=20,
        help="Number of validation batches to evaluate over.",
    )
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", 32)))
    return parser.parse_args()


def load_model(checkpoint_path: str) -> DeepPermNetViT:
    """Load model from a .pth (state-dict) or .ckpt (Lightning) file."""
    if checkpoint_path.endswith(".ckpt"):
        model = DeepPermNetViT.load_from_checkpoint(checkpoint_path)
        print(f"Loaded Lightning checkpoint: {checkpoint_path}")
    else:
        model = DeepPermNetViT()
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded state-dict: {checkpoint_path}")
    return model


def find_latest_checkpoint(save_dir: str) -> str:
    """Return the path of the most recently modified checkpoint in save_dir."""
    candidates = [
        os.path.join(save_dir, f)
        for f in os.listdir(save_dir)
        if f.endswith(".pth") or f.endswith(".ckpt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in {save_dir}")
    return max(candidates, key=os.path.getmtime)


def evaluate(model: DeepPermNetViT, val_loader, num_batches: int, device: torch.device):
    """
    Re-shuffle validation images on the fly (so the model sees fresh examples),
    run inference, and compute patch-level top-1 accuracy.
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (old_patches, old_perm) in enumerate(val_loader):
            if batch_idx >= num_batches:
                break

            B = old_patches.size(0)
            new_patches = torch.empty_like(old_patches)
            new_perm    = torch.empty_like(old_perm)

            # Re-shuffle each image in the batch independently
            for b in range(B):
                ordered = torch.empty_like(old_patches[b])
                ordered[old_perm[b]] = old_patches[b]          # un-shuffle first
                p = torch.randperm(9)
                new_patches[b] = ordered[p]
                new_perm[b]    = p

            new_patches = new_patches.to(device)
            new_perm    = new_perm.to(device)

            predicted_matrix  = model(new_patches)              # (B, 9, 9)
            predicted_indices = torch.argmax(predicted_matrix, dim=2)  # (B, 9)

            correct += (predicted_indices == new_perm).sum().item()
            total   += new_perm.numel()

    accuracy = 100.0 * correct / total
    return accuracy


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve checkpoint
    # ------------------------------------------------------------------
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = find_latest_checkpoint(args.save_dir)

    # ------------------------------------------------------------------
    # Data & model
    # ------------------------------------------------------------------
    data_module = DeepPermNetDataModule(batch_size=args.batch_size)
    data_module.setup()
    val_loader = data_module.val_dataloader()

    model = load_model(ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print(f"Evaluating over {args.num_batches} batches (batch_size={args.batch_size}) …")
    accuracy = evaluate(model, val_loader, args.num_batches, device)
    print(f"\nPatch-level accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
