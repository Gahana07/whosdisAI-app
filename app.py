import streamlit as st
import torch
from efficientnet_pytorch import model as eff_model
from PIL import Image
import io
import numpy as np
from torchvision import transforms
import cv2

# ----------------- Model Loading -----------------
@st.cache_resource
def load_model(model_path):
    # Allow EfficientNet for safe unpickling
    torch.serialization.add_safe_globals([eff_model.EfficientNet])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the full model object
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    
    return model, device

# ----------------- Image Preprocessing -----------------
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ----------------- Video Frame Extraction -----------------
def extract_frames(video_bytes, num_frames=30):
    frames = []
    video = cv2.VideoCapture(video_bytes)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    for i in range(0, total_frames, step):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        frames.append(frame_pil)
    video.release()
    return frames

# ----------------- Inference -----------------
def predict_image(model, device, image: Image.Image):
    img_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        prob = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
        confidence = prob[0, pred_class].item()
    return pred_class, confidence

# ----------------- Streamlit App -----------------
st.title("WhosDisAI - Deepfake Detector")

# Load model
model_path = "deepfake_model_full.pth"
model, device = load_model(model_path)

option = st.selectbox("Select input type", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            pred_class, confidence = predict_image(model, device, image)
            label = "FAKE" if pred_class == 1 else "REAL"
            st.success(f"Prediction: {label} ({confidence*100:.2f}%)")

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file:
        bytes_data = uploaded_file.read()
        with open("temp_video.mp4", "wb") as f:
            f.write(bytes_data)

        if st.button("Predict"):
            frames = extract_frames("temp_video.mp4", num_frames=30)
            results = [predict_image(model, device, frame) for frame in frames]
            fake_count = sum(1 for c, _ in results if c == 1)
            real_count = len(results) - fake_count
            label = "FAKE" if fake_count > real_count else "REAL"
            st.success(f"Video Prediction: {label} ({fake_count}/{len(results)} frames detected as FAKE)")

