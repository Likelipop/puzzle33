"""
visualize.py — Visual comparison of shuffled input, model prediction, and ground truth

Usage:
    python src/visualize.py --checkpoint ./models/ViT/deep_perm_net.pth [--num-samples 7]
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from dotenv import load_dotenv

from dataset import DeepPermNetDataModule
from model import DeepPermNetViT

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize DeepPermNet-ViT predictions")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.getenv("MODEL_SAVE_DIR", "./models/ViT"),
    )
    parser.add_argument("--num-samples", type=int, default=7)
    parser.add_argument("--batch-size",  type=int, default=int(os.getenv("BATCH_SIZE", 32)))
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the figure (e.g. results.png). Shows interactively if not given.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str) -> DeepPermNetViT:
    if checkpoint_path.endswith(".ckpt"):
        return DeepPermNetViT.load_from_checkpoint(checkpoint_path)
    model = DeepPermNetViT()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model


def find_latest_checkpoint(save_dir: str) -> str:
    candidates = [
        os.path.join(save_dir, f)
        for f in os.listdir(save_dir)
        if f.endswith(".pth") or f.endswith(".ckpt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in {save_dir}")
    return max(candidates, key=os.path.getmtime)


def patches_to_image(patches: torch.Tensor, indices: torch.Tensor):
    """Reorder 9 patches according to *indices* and stitch into a numpy image."""
    ordered = torch.zeros_like(patches)
    assigned = set()
    for cur, tgt in enumerate(indices.tolist()):
        if tgt not in assigned:
            ordered[tgt] = patches[cur]
            assigned.add(tgt)
    grid = make_grid(ordered, nrow=3, padding=2, pad_value=1.0)
    return grid.permute(1, 2, 0).cpu().numpy()


def visualize(model, val_loader, num_samples: int, output_path=None):
    model.eval()

    patches_batch, perm_batch = next(iter(val_loader))
    patches_batch = patches_batch[:num_samples]
    perm_batch    = perm_batch[:num_samples]

    with torch.no_grad():
        pred_matrix  = model(patches_batch)
        pred_indices = torch.argmax(pred_matrix, dim=2)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3.5 * num_samples))
    col_titles = ["1. Input (Shuffled)", "2. Model Prediction", "3. Ground Truth"]

    for i in range(num_samples):
        patches = patches_batch[i]
        seqs    = [torch.arange(9), pred_indices[i], perm_batch[i]]

        for col, seq in enumerate(seqs):
            axes[i, col].imshow(patches_to_image(patches, seq))
            axes[i, col].axis("off")
            if i == 0:
                axes[i, col].set_title(col_titles[col], fontsize=15, fontweight="bold")

    plt.suptitle("DeepPermNet-ViT — Patch Reassembly", fontsize=17, y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved → {output_path}")
    else:
        plt.show()


def main():
    args = parse_args()

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.save_dir)
    print(f"Using checkpoint: {ckpt_path}")

    data_module = DeepPermNetDataModule(batch_size=max(args.num_samples, args.batch_size))
    data_module.setup()
    val_loader = data_module.val_dataloader()

    model = load_model(ckpt_path)
    visualize(model, val_loader, args.num_samples, output_path=args.output)


if __name__ == "__main__":
    main()
