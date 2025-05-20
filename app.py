import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(
    page_title="Guess That Dog Breed!",
    page_icon="🐶",
    layout="centered"
)

# --------------------- CUSTOM STYLING ---------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #4CAF50;
    }
    .uploadbox {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        background-color: #ffffffdd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------- HEADER ---------------------
st.markdown("<h1 style='text-align: center;'>🐾 Guess That Dog Breed! 🐾</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a dog photo and let AI guess the breed!</p>", unsafe_allow_html=True)

# --------------------- LOAD MODEL & CLASS NAMES ---------------------
@st.cache(allow_output_mutation=True)
def load_model_and_classes():
    model = tf.keras.models.load_model("dog_breed_model.h5")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model_and_classes()

# --------------------- PREDICTION FUNCTION ---------------------
def predict_breed(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_breed = class_names[predicted_index]
    return predicted_breed, confidence

# --------------------- UPLOAD SECTION ---------------------
st.markdown("<div class='uploadbox'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a dog image", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded Image", use_column_width=True)

    with st.spinner('Analyzing image...'):
        breed, conf = predict_breed(image_display)
        st.success(f"🎉 **Predicted Breed:** `{breed}`")
        st.info(f"🔍 **Confidence:** `{conf:.2f}`")

# --------------------- FOOTER ---------------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 0.9em;'>Made with ❤️ using TensorFlow & Streamlit by TZ</p>
    """,
    unsafe_allow_html=True
)
