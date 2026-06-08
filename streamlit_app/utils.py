import torch
import streamlit as st
from torchvision import transforms
from scipy.optimize import linear_sum_assignment

def sinkhorn(matrix, n_iters=20):
    """Applies the Sinkhorn algorithm to output a soft permutation matrix."""
    P = torch.exp(matrix)
    for _ in range(n_iters):
        P = P / P.sum(dim=2, keepdim=True)
        P = P / P.sum(dim=1, keepdim=True)
    return P

def create_grid_image(patches, perm):
    """Reconstructs the image grid based on patches and a given permutation."""
    canvas = torch.zeros(3, 192, 192)
    for j, slot in enumerate(perm):
        r, c = int(slot) // 3, int(slot) % 3
        canvas[:, r*64:(r+1)*64, c*64:(c+1)*64] = patches[j]
    return canvas.clamp(0, 1).permute(1, 2, 0).numpy()

@st.cache_data
def get_img_transform():
    """Returns the standard image transformation pipeline."""
    return transforms.Compose([
        # Force the image to be exactly 192x192. No cropping, no lost edges.
        transforms.Resize((192, 192)), 
        transforms.ToTensor(),
    ])

def solve_permutation(raw_preds):
    """Takes raw model predictions and returns the final assigned indices."""
    sinkhorn_probs = sinkhorn(raw_preds, n_iters=20)
    cost_matrix = -sinkhorn_probs[0].numpy() 
    _, col_ind = linear_sum_assignment(cost_matrix)
    return torch.tensor(col_ind)