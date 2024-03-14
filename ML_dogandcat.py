import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def load_model():
    model = tf.keras.models.load_model('C:\\Users\\debanandan\\Downloads\\cat dog\\Final Result\\ML_dogcat.h5')  # Replace 'model.h5' with the path to your saved model
    return model

def preprocess_image(image):
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0
    return image

model = load_model()

st.title('Cat vs Dog Classifier')


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    image = np.array(image)
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    class_names = ['Cat', 'Dog']  
    predicted_class = class_names[int(round(prediction[0][0]))]
    st.write(f"Prediction: {predicted_class}")
    prediction = model.predict(image)
    st.write(f"Raw Prediction: {prediction}")

