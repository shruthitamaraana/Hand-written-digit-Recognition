import streamlit as st
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Set model path
MODEL_PATH = "models/mnist_model.keras"

@st.cache_resource
def load_model():
    """Load the trained MNIST model."""
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error("❌ Model file not found! Please check the model path.")
        st.stop()

# Load the model once (cached)
model = load_model()
st.success("✅ Model loaded successfully!")

# Streamlit app title
st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload an image or draw a digit (28x28 pixels), and the model will predict it.")

# Sidebar for mode selection
option = st.sidebar.radio("Choose Input Method", ("Upload Image", "Draw Digit"))

def preprocess_image(image):
    """Convert image to grayscale, resize, normalize, and reshape for the model."""
    image = image.convert('L').resize((28, 28))  # Convert to grayscale and resize
    image = np.array(image).astype('float32') / 255  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

def predict_digit(image):
    """Predict the digit using the trained model."""
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit, prediction

# Handling file upload
if option == "Upload Image":
    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"], key="file_uploader")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
        
        image = preprocess_image(Image.open(uploaded_file))
        predicted_digit, prediction = predict_digit(image)

        st.success(f"🟢 Predicted Digit: **{predicted_digit}**")
        st.bar_chart(prediction[0])  # Show confidence scores

# Handling drawing on canvas
elif option == "Draw Digit":
    st.write("✏️ Draw your digit below and click **Predict**!")

    canvas_result = st_canvas(
        stroke_width=10, stroke_color="white", background_color="black",
        width=280, height=280, key="canvas"
    )

    if st.button("🔍 Predict", key="predict_button"):
        if canvas_result.image_data is not None:
            image = Image.fromarray((canvas_result.image_data[..., :3]).astype('uint8'))
            image = preprocess_image(image)

            predicted_digit, prediction = predict_digit(image)

            st.success(f"🟢 Predicted Digit: **{predicted_digit}**")
            st.bar_chart(prediction[0])  # Show confidence scores

# Footer
st.markdown("---")
st.write("🚀 Developed with ❤️ using Streamlit and TensorFlow")
