import torch
import onnxruntime as ort
import streamlit as st

@st.cache_resource
def load_model():
    """Loads and caches the ONNX inference session."""
    return ort.InferenceSession("models/ViT/deep_perm_net.onnx")

def run_onnx_inference(session, inputs):
    """Runs a batch of inputs through the ONNX session."""
    outputs = []
    for i in range(inputs.size(0)):
        single = inputs[i:i+1].cpu().numpy()
        ort_in = {session.get_inputs()[0].name: single}
        outputs.append(torch.tensor(session.run(None, ort_in)[0]))
    return torch.cat(outputs, dim=0)