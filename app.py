import streamlit as st
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Ensure correct model path
MODEL_PATH = os.path.join(os.getcwd(), "models", "mnist_model.keras")

# Load the model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Model file not found! Please check the model path.")

# Streamlit app title
st.title("🖊️ Handwritten Digit Recognition")
st.write("Upload an image or draw a digit (28x28 pixels), and the model will predict it.")

# Sidebar for mode selection
option = st.sidebar.radio("Choose Input Method", ("Upload Image", "Draw Digit"))

# Preprocess image function
def preprocess_image(image):
    """Convert image to grayscale, resize, normalize, and reshape for the model."""
    image = image.convert('L').resize((28, 28))  # Convert to grayscale and resize
    image = np.array(image).astype('float32') / 255  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

# Prediction function
def predict_digit(image):
    """Predict the digit and return probabilities."""
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit, prediction

# 📂 **Option 1: Upload Image**
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        image = preprocess_image(Image.open(uploaded_file))
        predicted_digit, prediction = predict_digit(image)
        
        # Display results
        st.success(f"🎯 Predicted Digit: **{predicted_digit}**")
        st.bar_chart(prediction[0])  # Show confidence scores

# ✍️ **Option 2: Draw Digit**
elif option == "Draw Digit":
    st.write("Draw your digit below and click **Predict**!")

    # Canvas for drawing
    canvas_result = st_canvas(
        stroke_width=10, stroke_color="white", background_color="black",
        width=280, height=280, key="canvas"
    )

    if st.button("Predict"):
        if canvas_result.image_data is not None:
            # Convert canvas drawing to image
            image = Image.fromarray((canvas_result.image_data[..., :3]).astype('uint8'))
            image = preprocess_image(image)

            # Predict the digit
            predicted_digit, prediction = predict_digit(image)

            # Display results
            st.success(f"🎯 Predicted Digit: **{predicted_digit}**")
            st.bar_chart(prediction[0])  # Show confidence scores

# Footer
st.markdown("---")
st.write("🔗 Developed with ❤️ using Streamlit and TensorFlow")
