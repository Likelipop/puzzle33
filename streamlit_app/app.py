import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import os

# Set page title and structure
st.set_page_config(page_title="DeepPermNet Puzzle Solver", layout="centered")
st.title("🧩 DeepPermNet: Visual Permutation Solver")
st.write("Upload an image containing a shuffled 3x3 grid layout to restore its original visual structure.")

# Define model path
MODEL_PATH = "models/ViT/deep_perm_net.onnx"

@st.cache_resource
def load_onnx_session(model_path):
    """Safely initializes and caches the inference session."""
    if not os.path.exists(model_path):
        st.error(f"ONNX model file not found at: `{model_path}`. Please verify your folder hierarchy.")
        return None
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

session = load_onnx_session(MODEL_PATH)

def process_and_slice_image(uploaded_file):
    """
    Prevents Preprocessing Drift by mirroring the training patch sequence logic explicitly.
    Forces bilinear downsampling, layout consistency, and standard token shape configurations.
    """
    # 1. Enforce strict baseline format conversion (RGB) to eliminate channel count drift
    img = Image.open(uploaded_file).convert("RGB")
    
    # 2. Match notebook spatial target bounds exactly (3 patches * 64px = 192px width/height)
    target_size = 192
    img_resized = img.resize((target_size, target_size), resample=Image.Resampling.BILINEAR)
    
    # Convert to pure float32 numpy array normalized to [0.0, 1.0] range
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Extract structural patch dimensions
    patch_size = 64
    patches = []
    
    # 3. Slice sequentially into a 3x3 grid matching training positional indexing
    for i in range(3):
        for j in range(3):
            y_start = i * patch_size
            x_start = j * patch_size
            patch = img_np[y_start:y_start+patch_size, x_start:x_start+patch_size, :]
            
            # Rearrange channel dimensions from (H, W, C) to (C, H, W) to match model expectations
            patch = np.transpose(patch, (2, 0, 1))
            patches.append(patch)
            
    # Pack array tokens into sequence format: (9, 3, 64, 64)
    patches_seq = np.array(patches, dtype=np.float32)
    
    # Add fake batch size dimension to satisfy ONNX input criteria: (1, 9, 3, 64, 64)
    input_tensor = np.expand_dims(patches_seq, axis=0)
    return input_tensor, img_resized

def reassemble_image(original_resized, predicted_matrix):
    """
    Utilizes Hungarian argmax matching criteria to map indices back to standard matrix coordinates.
    """
    img_np = np.array(original_resized)
    patch_size = 64
    
    # Initialize blank canvas for structural stitching
    reconstructed = np.zeros_like(img_np)
    
    # Pull sequence index mappings via argmax logic on the assignment layer
    # predicted_matrix shape: [1, 9, 9] -> access index 0
    mapping_matrix = predicted_matrix[0]
    assigned_indices = np.argmax(mapping_matrix, axis=1)
    
    # Tracking references to avoid position duplicate overlays
    used_positions = set()
    
    # Define reference coordinates for standard linear order layout mapping
    grid_coords = [(i * patch_size, j * patch_size) for i in range(3) for j in range(3)]
    
    for current_idx, target_idx in enumerate(assigned_indices):
        if target_idx not in used_positions:
            used_positions.add(target_idx)
            
            # Determine coordinate source vs destination mappings
            src_y, src_x = grid_coords[current_idx]
            dest_y, dest_x = grid_coords[target_idx]
            
            # Cut patch piece from current shuffled coordinates and place into target original layer
            reconstructed[dest_y:dest_y+patch_size, dest_x:dest_x+patch_size, :] = \
                img_np[src_y:src_y+patch_size, src_x:src_x+patch_size, :]
        else:
            # Fallback for handling collisions or duplicate predictions gracefully
            src_y, src_x = grid_coords[current_idx]
            reconstructed[src_y:src_y+patch_size, src_x:src_x+patch_size, :] = \
                img_np[src_y:src_y+patch_size, src_x:src_x+patch_size, :]
                
    return Image.fromarray(reconstructed)

# UI Implementation
uploaded_file = st.file_uploader("Choose a shuffled layout image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and session is not None:
    # Render operational layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Shuffled Input Image")
        st.image(uploaded_file, use_column_width=True)
        
    with st.spinner("Extracting patches and generating permutation graph..."):
        try:
            # Execute drift-proof feature engineering pipeline
            input_tensor, processed_img = process_and_slice_image(uploaded_file)
            
            # Fetch default input signature parameters dynamically
            input_name = session.get_inputs()[0].name
            
            # Run inference over ONNX backend runtime architecture
            outputs = session.run(None, {input_name: input_tensor})
            predicted_permutation = outputs[0]  # Array representing doubly-stochastic assignments
            
            # Solve positional tracking matrix and re-stitch visual structure
            restored_image = reassemble_image(processed_img, predicted_permutation)
            
            with col2:
                st.subheader("Predicted Groundtruth Restoration")
                st.image(restored_image, use_column_width=True)
                
            st.success("🎉 Permutation matrix solved successfully!")
            
        except Exception as e:
            st.error(f"Inference Failure Exception triggered: {str(e)}")
            st.info("Check if your ONNX engine layout layers correspond to the signature dimension: `(1, 9, 3, 64, 64)`.")