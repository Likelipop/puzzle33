import streamlit as st
import io
import random
import torch
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from scipy.optimize import linear_sum_assignment

st.set_page_config(page_title="DeepPermNet Solver", layout="wide", page_icon="🧩")

# ── Setup & Caching ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return ort.InferenceSession("models/ViT/deep_perm_net.onnx")

ort_session = load_model()

def sinkhorn(matrix, n_iters=20):
    P = torch.exp(matrix)
    for _ in range(n_iters):
        P = P / P.sum(dim=2, keepdim=True)
        P = P / P.sum(dim=1, keepdim=True)
    return P

def create_grid_image(patches, perm):
    canvas = torch.zeros(3, 192, 192)
    for j, slot in enumerate(perm):
        r, c = int(slot) // 3, int(slot) % 3
        canvas[:, r*64:(r+1)*64, c*64:(c+1)*64] = patches[j]
    return canvas.clamp(0, 1).permute(1, 2, 0).numpy()

def run_onnx_inference(session, inputs):
    outputs = []
    for i in range(inputs.size(0)):
        single = inputs[i:i+1].cpu().numpy()
        ort_in = {session.get_inputs()[0].name: single}
        outputs.append(torch.tensor(session.run(None, ort_in)[0]))
    return torch.cat(outputs, dim=0)

@st.cache_data
def get_img_transform():
    return transforms.Compose([
        # Force the image to be exactly 192x192. No cropping, no lost edges.
        transforms.Resize((192, 192)), 
        transforms.ToTensor(),
    ])

img_transform = get_img_transform()

# ── Application UI ─────────────────────────────────────────────────────────────
st.title("🧩 DeepPermNet: Visual Permutation Solver")

# --- NEW: ADD A TOGGLE FOR PRE-SHUFFLED IMAGES ---
is_pre_shuffled = st.toggle("Bypass Shuffle (Check this if the image you are uploading is ALREADY shuffled)", value=False)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_tensor = img_transform(pil_img)
    
    patches = [img_tensor[:, r*64:(r+1)*64, c*64:(c+1)*64] 
               for r in range(3) for c in range(3)]
               
    # --- MODIFIED LOGIC: Respect the toggle ---
    if is_pre_shuffled:
        # Do NOT shuffle. Use the patches exactly as they were uploaded.
        shuffled_patches = torch.stack(patches)
        ground_truth = None # We don't have a ground truth score for this
    else:
        # Standard flow: Shuffle the image
        shuffled_indices = list(range(9))
        random.shuffle(shuffled_indices)
        ground_truth = torch.tensor(shuffled_indices)
        shuffled_patches = torch.stack([patches[i] for i in shuffled_indices])
    
    # Inference
    with st.spinner('Running ONNX inference...'):
        raw_preds = run_onnx_inference(ort_session, shuffled_patches.unsqueeze(0))
        sinkhorn_probs = sinkhorn(raw_preds, n_iters=20)
        
        cost_matrix = -sinkhorn_probs[0].numpy() 
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        preds = torch.tensor(col_ind)
        
    st.divider()
    
    # --- MODIFIED VISUALIZATION: Adapt to the toggle state ---
    if is_pre_shuffled:
        st.subheader("Results (Bypass Mode)")
        st.markdown("*Note: Because you uploaded an already-shuffled image, we cannot calculate an accuracy percentage. You are the judge!*")
        
        img_uploaded = create_grid_image(shuffled_patches, torch.arange(9))
        img_predicted = create_grid_image(shuffled_patches, preds)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### What you Uploaded")
            st.image(img_uploaded, use_container_width=True)
        with col2:
            st.markdown("#### Model's Attempt to Fix It")
            st.image(img_predicted, use_container_width=True)
            
    else:
        # Standard Display
        correct_patches = (preds == ground_truth).sum().item()
        accuracy = (correct_patches / 9) * 100
        st.metric(label="Reconstruction Accuracy", value=f"{correct_patches}/9", delta=f"{accuracy:.1f}%")
        
        img_original = create_grid_image(torch.stack(patches), torch.arange(9))
        img_shuffled = create_grid_image(shuffled_patches, torch.arange(9))
        img_predicted = create_grid_image(shuffled_patches, preds)
        img_ground_truth = create_grid_image(shuffled_patches, ground_truth)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("#### 1. Original")
            st.image(img_original, use_container_width=True)
        with col2:
            st.markdown("#### 2. Internal Shuffle")
            st.image(img_shuffled, use_container_width=True)
        with col3:
            st.markdown("#### 3. Predicted Fix")
            st.image(img_predicted, use_container_width=True)
        with col4:
            st.markdown("#### 4. Ground Truth")
            st.image(img_ground_truth, use_container_width=True)