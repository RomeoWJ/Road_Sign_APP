import streamlit as st
import torch
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add yolov5 folder to path
sys.path.append(str(Path(__file__).parent / 'yolov5'))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

@st.cache_resource
def load_model():
    model = attempt_load('best.pt', map_location='cpu')
    return model

model = load_model()

st.title("🚦 Road Sign Detection App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = letterbox(np.array(image), 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, RGB to BGR
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    # Display detections
    if pred is not None and len(pred):
        st.write(f"Detections: {len(pred)} object(s) found.")
    else:
        st.write("No detections found.")

    st.success("Detection Complete!")
