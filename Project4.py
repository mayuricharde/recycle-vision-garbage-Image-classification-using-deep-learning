import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="RecycleVision",
    page_icon="♻️",
    layout="centered"
)


# -----------------------------
# Model Path
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "final_garbage_classifier.keras"


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# -----------------------------
# Class Names
# -----------------------------
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# -----------------------------
# Title
# -----------------------------
st.title("♻️ RecycleVision")
st.write("Garbage Classification using Deep Learning")


# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a garbage image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # -----------------------------
    # Preprocessing
    # -----------------------------
    resized_image = image.resize((224, 224))

    image_array = np.array(resized_image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_class = np.argmax(prediction[0])

    confidence = prediction[0][predicted_class] * 100


    # -----------------------------
    # Display Result
    # -----------------------------
    st.subheader("Prediction")

    st.success(
        f"♻️ Category: {class_names[predicted_class].title()}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )


    # -----------------------------
    # All Class Probabilities
    # -----------------------------
    st.subheader("Class Probabilities")

    for i, class_name in enumerate(class_names):

        probability = prediction[0][i] * 100

        st.write(
            f"**{class_name.title()}**: {probability:.2f}%"
        )

        st.progress(
            float(prediction[0][i])
        )