import streamlit as st
import random
import torch
from PIL import Image

# Import from our new modules
from model import load_model, run_onnx_inference
from utils import get_img_transform, create_grid_image, solve_permutation

st.set_page_config(page_title="DeepPermNet Solver", layout="wide", page_icon="🧩")

# Load model and transforms
ort_session = load_model()
img_transform = get_img_transform()

# ── Application UI ─────────────────────────────────────────────────────────────
st.title("🧩 DeepPermNet: Visual Permutation Solver")

is_pre_shuffled = st.toggle("Bypass Shuffle (Check this if the image you are uploading is ALREADY shuffled)", value=False)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_tensor = img_transform(pil_img)
    
    # Extract 3x3 patches
    patches = [img_tensor[:, r*64:(r+1)*64, c*64:(c+1)*64] 
               for r in range(3) for c in range(3)]
               
    if is_pre_shuffled:
        shuffled_patches = torch.stack(patches)
        ground_truth = None
    else:
        shuffled_indices = list(range(9))
        random.shuffle(shuffled_indices)
        ground_truth = torch.tensor(shuffled_indices)
        shuffled_patches = torch.stack([patches[i] for i in shuffled_indices])
    
    # Inference
    with st.spinner('Running ONNX inference...'):
        raw_preds = run_onnx_inference(ort_session, shuffled_patches.unsqueeze(0))
        preds = solve_permutation(raw_preds)
        
    st.divider()
    
    # Visualization
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