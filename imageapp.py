import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('models/imageclassifier.h5')

# Title of the Streamlit app
st.title("😊 Happy vs 😢 Sad Image Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)       # Convert to RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize and normalize
    resized_img = tf.image.resize(image, (256, 256))
    normalized_img = resized_img / 255.0
    input_tensor = np.expand_dims(normalized_img, axis=0)

    # Make prediction
    yhat = model.predict(input_tensor)

    # Display prediction
    prediction = "Sad 😢" if yhat > 0.5 else "Happy 😊"
    st.subheader(f"Prediction: {prediction}")
    st.caption(f"Confidence: {yhat[0][0]:.4f}")
