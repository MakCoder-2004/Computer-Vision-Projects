import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="B&W Image Colorizer",
    page_icon="🎨",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 1000px;
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    .stFileUploader {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎨 Black & White Image Colorizer")
st.markdown("Upload a black and white image to see it come to life with AI-powered colorization!")

# File uploader
uploaded_file = st.file_uploader("Choose a black and white image...", type=["jpg", "jpeg", "png"])

# Initialize session state to store the colorized image
if 'colorized_image' not in st.session_state:
    st.session_state.colorized_image = None

# Function to load the colorization model
def load_colorizer():
    # Load pre-trained model and points
    prototxt_path = os.path.join("..", "pretrained model", "colorization_deploy_v2.prototxt")
    model_path = os.path.join("..", "pretrained model", "colorization_release_v2.caffemodel")
    kernel_path = os.path.join("..", "points", "pts_in_hull.npy")
    
    # Load the model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)
    
    # Prepare the network
    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]
    
    return net

# Function to colorize the image
def colorize_image(net, input_image):
    # Convert to numpy array
    image = np.array(input_image)
    
    # Convert to BGR (OpenCV format)
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert to float32 and normalize
    normalized = image.astype(np.float32) / 255.0
    
    # Convert to LAB color space
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    
    # Resize and extract L channel
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L = L - 50  # Mean centering
    
    # Set the input and get the predicted 'ab' channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Resize the predicted 'ab' channels to match the original image size
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    
    # Get the L channel from the original image
    L = cv2.split(lab)[0]
    
    # Combine the L channel with the predicted 'ab' channels
    colorized_lab = cv2.merge([L, ab])
    
    # Convert back to BGR color space and scale to 0-255
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr * 255, 0, 255).astype(np.uint8)
    
    # Convert back to RGB for display
    colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
    
    return colorized_rgb

# Main app logic
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Colorized Image")
        
        # Show a loading spinner while processing
        with st.spinner('Colorizing image... This may take a moment...'):
            # Load the model (only once)
            if 'net' not in st.session_state:
                st.session_state.net = load_colorizer()
            
            # Colorize the image
            colorized = colorize_image(st.session_state.net, image)
            st.session_state.colorized_image = colorized
            
            # Display the colorized image
            st.image(colorized, use_container_width=True)
            
            # Add a download button
            buffered = io.BytesIO()
            Image.fromarray(colorized).save(buffered, format="JPEG")
            st.download_button(
                label="Download Colorized Image",
                data=buffered.getvalue(),
                file_name="colorized_image.jpg",
                mime="image/jpeg"
            )


# Add a footer
st.markdown("---")
st.markdown("*Powered by OpenCV and Streamlit*")

