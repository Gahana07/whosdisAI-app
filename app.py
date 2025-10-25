# app.py
import streamlit as st
from PIL import Image
import os

st.title("whosdisAI - Deepfake Detection")

st.write("""
Upload an image or video and our AI model will detect if it's real or fake!
""")

# Upload file
uploaded_file = st.file_uploader("Choose an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    file_name = uploaded_file.name

    st.write(f"File uploaded: {file_name}")

    # If it's an image
    if "image" in file_type:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Detection result: [This is where your model prediction will appear]")

    # If it's a video
    elif "video" in file_type:
        video_bytes = uploaded_file.read()
        st.video(video_bytes)
        st.write("Detection result: [This is where your model prediction will appear]")

st.write("Developed by Gahana07")
