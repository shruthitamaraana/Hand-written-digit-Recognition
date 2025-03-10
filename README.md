# Handwritten Digit Recognition

# 📌 Overview

This is a web application for recognizing handwritten digits (0-9) using a deep learning model trained on the MNIST dataset. Users can either upload an image of a digit or draw a digit directly on the canvas, and the model will predict the digit.

# 🚀 Features

- Upload an image (JPG, PNG, JPEG) for digit recognition
- Draw a digit on a canvas and get predictions
- Uses a deep learning model trained with TensorFlow/Keras
- Streamlit-powered interactive web app

# 🛠️ Technologies Used

- **Python**
- **TensorFlow/Keras**
- **NumPy**
- **PIL (Pillow)**
- **Streamlit**
- **Streamlit Drawable Canvas**

# 🏗️ Project Structure

```
handwritten-digit-recognition/
│-- models/
│   ├── mnist_model.keras  # Trained model file
│-- app.py                 # Streamlit app code
│-- train_model.py         # Model training script
│-- requirements.txt       # Dependencies list
│-- README.md              # Project documentation
```

#📥 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shruthitamaraana/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

# 🚀 Running the Application

```bash
streamlit run app.py
```

This will launch the application in your web browser.

# 🌐 Deploying to Streamlit Cloud

1. Push all project files to GitHub.
2. Go to **[Streamlit Cloud](https://share.streamlit.io/)** and log in.
3. Click on **New App** → Select your GitHub repository.
4. Set the **main file path** as `app.py`.
5. Click **Deploy**!

# 🔥 Example Usage

1. Upload a handwritten digit image or draw a digit.
2. Click **Predict**.
3. The app will display the predicted digit along with confidence scores.

# 🤖 Model Training

If you want to train the model yourself, run:

```bash
python train_model.py
```
# 🚀 Live Demo
👉 [Click here to open the app]([https://your-username-your-repo-name.streamlit.](https://hand-written-digit-recognition-8ckmd6fkglcsrtnk4s4nn5.streamlit.app/))

https://hand-written-digit-recognition-8ckmd6fkglcsrtnk4s4nn5.streamlit.app/

📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

Made with ❤️ by Shruthi Tamaraana
