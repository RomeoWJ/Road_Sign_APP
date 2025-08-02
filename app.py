<<<<<<< HEAD
import torch
from pathlib import Path
import streamlit as st
from PIL import Image

@st.cache_resource
def load_model():
    import zipfile
    import os
    import torch

    # Download yolov5 zip to /tmp
    if not os.path.exists('/tmp/yolov5'):
        zip_path = '/tmp/yolov5.zip'
        torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip', zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('/tmp/')

        # Rename extracted folder to /tmp/yolov5
        os.rename('/tmp/yolov5-master', '/tmp/yolov5')

    model = torch.hub.load('/tmp/yolov5', 'custom', path='best.pt', source='local')
=======
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
>>>>>>> 7ddbe00 (Use local yolov5 repo for detection)
    return model


model = load_model()

<<<<<<< HEAD
# Rest of your Streamlit UI remains the same
st.title("🚦 Road Sign Detection App (Custom Model)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting..."):
        results = model(image)
        results.render()

    st.image(results.ims[0], caption="Detection Result", use_column_width=True)
=======
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

>>>>>>> 7ddbe00 (Use local yolov5 repo for detection)
    st.success("Detection Complete!")
