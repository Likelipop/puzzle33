"""
model.py — DeepPermNet-ViT Architecture

Components:
  - sinkhorn_operator: differentiable doubly-stochastic approximation
  - DeepPermNetViT: Siamese CNN + Transformer Encoder + Sinkhorn output head
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Sinkhorn Layer
# ---------------------------------------------------------------------------

def sinkhorn_operator(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    Applies Sinkhorn iterations to turn a raw (batch, N, N) matrix into a
    doubly-stochastic matrix (each row and column sums to 1).

    Args:
        log_alpha: Tensor of shape (B, N, N) — raw logits from the prediction head.
        n_iters:   Number of alternating row/column normalisation steps.

    Returns:
        Doubly-stochastic matrix of shape (B, N, N).
    """
    P = torch.exp(log_alpha)
    for _ in range(n_iters):
        P = P / P.sum(dim=2, keepdim=True)   # row normalise
        P = P / P.sum(dim=1, keepdim=True)   # column normalise
    return P


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeepPermNetViT(pl.LightningModule):
    """
    DeepPermNet with a Vision-Transformer backbone.

    Architecture:
      1. Siamese CNN feature extractor  — processes each of the 9 patches
         independently, sharing weights (output: 2048-d vector per patch).
      2. Token projection               — linear 2048 → d_model.
      3. Transformer Encoder            — captures inter-patch relationships.
      4. Prediction head                — linear d_model → 9 (raw logits).
      5. Sinkhorn layer                 — converts logits to a doubly-stochastic
         permutation matrix.

    Loss: MSE between the predicted doubly-stochastic matrix and the
          ground-truth one-hot permutation matrix.
    """

    def __init__(
        self,
        learning_rate: float = None,
        d_model: int = None,
        nhead: int = None,
        num_layers: int = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate or float(os.getenv("LEARNING_RATE", 1e-4))
        d_model     = d_model     or int(os.getenv("D_MODEL",     512))
        nhead       = nhead       or int(os.getenv("NHEAD",       8))
        num_layers  = num_layers  or int(os.getenv("NUM_LAYERS",  4))

        self.save_hyperparameters()

        # --- 1. Siamese CNN ---
        # Input: (B*9, 3, 64, 64)  →  output: (B*9, 2048)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,   32,  kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),   # 128 × (64/16)² = 128 × 16 = 2048
        )

        # --- 2. Projection ---
        self.token_projection = nn.Linear(2048, d_model)

        # --- 3. Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- 4. Prediction head ---
        self.prediction_head = nn.Linear(d_model, 9)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 9, 3, 64, 64) — batch of 9-patch sequences.
        Returns:
            Doubly-stochastic matrix of shape (B, 9, 9).
        """
        B, S, C, H, W = x.size()

        # Extract features for all patches simultaneously
        cnn_out = self.feature_extractor(x.view(B * S, C, H, W))  # (B*9, 2048)
        tokens  = cnn_out.view(B, S, -1)                           # (B, 9, 2048)
        tokens  = self.token_projection(tokens)                    # (B, 9, d_model)

        # Self-attention across patch tokens
        ctx = self.transformer(tokens)                             # (B, 9, d_model)

        # Raw 9×9 logits → Sinkhorn approximation
        logits = self.prediction_head(ctx)                         # (B, 9, 9)
        return sinkhorn_operator(logits)

    # ------------------------------------------------------------------
    def _loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_matrix = F.one_hot(target, num_classes=9).float()   # (B, 9, 9)
        return F.mse_loss(predicted, target_matrix)

    def training_step(self, batch, batch_idx):
        patches, perm = batch
        pred = self(patches)
        loss = self._loss(pred, perm)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patches, perm = batch
        pred = self(patches)
        loss = self._loss(pred, perm)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
