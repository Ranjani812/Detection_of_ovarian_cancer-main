import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Ovarian Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 500;
    }
    .result-text {
        font-size: 1.8rem;
        font-weight: 600;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E3F2FD;
        border-left: 0.5rem solid #1E88E5;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.markdown('<p class="main-header">Ovarian Cancer Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Upload ultrasound images to detect and classify ovarian cancer subtypes using deep learning</p>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

# Function to load the model
@st.cache_resource
def load_ml_model():
    try:
        model = load_model('my_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to make predictions
def predict_image(img, model):
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    return prediction

# Class names and their full forms
class_names = ['cc', 'ec', 'hgsc', 'lgsc', 'mc']
class_full_names = {
    'cc': 'Clear Cell Carcinoma',
    'ec': 'Endometrioid Carcinoma',
    'hgsc': 'High-Grade Serous Carcinoma',
    'lgsc': 'Low-Grade Serous Carcinoma',
    'mc': 'Mucinous Carcinoma'
}

# Class descriptions for educational purposes
class_descriptions = {
    'cc': 'Clear cell carcinoma is characterized by cells with clear cytoplasm and is often associated with endometriosis.',
    'ec': 'Endometrioid carcinoma resembles the endometrium (lining of the uterus) and is typically less aggressive.',
    'hgsc': 'High-grade serous carcinoma is the most common and aggressive subtype, often diagnosed at an advanced stage.',
    'lgsc': 'Low-grade serous carcinoma is less common and typically has a better prognosis than HGSC.',
    'mc': 'Mucinous carcinoma contains cells that secrete mucin and is often diagnosed at an early stage.'
}

# Sidebar with information
with st.sidebar:
    st.markdown('<p class="sub-header">About</p>', unsafe_allow_html=True)
    st.info("This application uses deep learning to detect and classify different types of ovarian cancer from ultrasound images.")
    
    st.markdown('<p class="sub-header">Cancer Subtypes</p>', unsafe_allow_html=True)
    for cls, full_name in class_full_names.items():
        with st.expander(f"{full_name} ({cls})"):
            st.write(class_descriptions[cls])
    
    st.markdown('<p class="sub-header">Instructions</p>', unsafe_allow_html=True)
    st.write("1. Upload an ultrasound image")
    st.write("2. Click on 'Detect Cancer Type'")
    st.write("3. View the results and probability scores")

# Load the model
model = load_ml_model()

with col1:
    st.markdown('<p class="sub-header">Upload Ultrasound Image</p>', unsafe_allow_html=True)
    
    # Image upload widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Detect Cancer Type"):
            if model is not None:
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    prediction = predict_image(image_pil, model)
                    
                    # Get predicted class
                    predicted_class_index = np.argmax(prediction[0])
                    predicted_class = class_names[predicted_class_index]
                    confidence = float(prediction[0][predicted_class_index]) * 100
                    
                    # Switch to the results column
                    with col2:
                        st.markdown('<p class="sub-header">Detection Results</p>', unsafe_allow_html=True)
                        
                        st.markdown(f'<div class="success-box">'
                                   f'<p class="result-text">Detected: {class_full_names[predicted_class]}</p>'
                                   f'<p>Confidence: {confidence:.2f}%</p>'
                                   f'</div>', unsafe_allow_html=True)
                        
                        # Create a bar chart for all class probabilities
                        fig, ax = plt.subplots(figsize=(10, 6))
                        y_pos = np.arange(len(class_names))
                        
                        # Sort probabilities for better visualization
                        sorted_indices = np.argsort(prediction[0])[::-1]
                        sorted_classes = [class_full_names[class_names[i]] for i in sorted_indices]
                        sorted_probs = [prediction[0][i] * 100 for i in sorted_indices]
                        
                        # Create bars with gradient colors
                        bars = ax.barh(y_pos, sorted_probs, align='center', color='skyblue')
                        bars[0].set_color('#1E88E5')  # Highlight the highest probability
                        
                        # Add labels and formatting
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(sorted_classes)
                        ax.invert_yaxis()
                        ax.set_xlabel('Probability (%)')
                        ax.set_title('Cancer Type Probability Distribution')
                        
                        # Add percentage labels to bars
                        for i, v in enumerate(sorted_probs):
                            ax.text(v + 1, i, f"{v:.1f}%", va='center')
                        
                        st.pyplot(fig)
                        
                        # Add description of the detected cancer type
                        st.markdown('<p class="sub-header">About This Cancer Type</p>', unsafe_allow_html=True)
                        st.info(class_descriptions[predicted_class])
                        
                        # Recommendations based on detected type
                        st.markdown('<p class="sub-header">Recommended Next Steps</p>', unsafe_allow_html=True)
                        st.write("• Consult with a gynecologic oncologist for a comprehensive evaluation")
                        st.write("• Consider additional diagnostic tests to confirm the findings")
                        st.write("• Discuss treatment options based on the cancer subtype and stage")
            else:
                st.error("Model could not be loaded. Please check the model file path.")

# Initial state for the second column
if uploaded_file is None:
    with col2:
        st.markdown('<p class="sub-header">Detection Results</p>', unsafe_allow_html=True)
        st.write("Upload an image and click 'Detect Cancer Type' to see results.")

# Footer
st.markdown('<p class="footer">This application is for educational and research purposes only. Always consult healthcare professionals for medical advice.</p>', unsafe_allow_html=True)