import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("deepfake_mobilenet.h5")
    return model

model = load_model()

st.title("Deepfake Detector 🔍")
st.write("Upload an image to check if it's REAL or FAKE.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Predict
    pred = model.predict(x)[0][0]

    st.write("### Prediction Score:", float(pred))

    if pred > 0.5:
        st.error("⚠️ FAKE (Deepfake detected!)")
    else:
        st.success("✅ REAL (No deepfake found)")
