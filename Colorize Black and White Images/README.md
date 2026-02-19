# Black & White Image Colorization using Deep Learning

## 🚀 Overview
This project brings old black and white photos to life using deep learning. It uses a pre-trained Caffe model to automatically colorize grayscale images with realistic colors.

## 🛠️ Technologies Used

- **OpenCV**: For image processing and computer vision tasks
- **Streamlit**: For creating the web interface
- **NumPy**: For numerical operations
- **Pillow (PIL)**: For image handling
- **Caffe**: Deep learning framework (using pre-trained model)

## 🏗️ Project Structure

```
Colorize Black and White Images/
├── colorize_image.py     # Core colorization script
├── streamlit/
│   └── app.py           # Streamlit web application
├── pretrained model/     # Pre-trained Caffe model files
│   ├── colorization_deploy_v2.prototxt
│   └── colorization_release_v2.caffemodel
└── points/               # Cluster center points for colorization
    └── pts_in_hull.npy
```

## 🚀 How It Works

The colorization process involves several steps:
1. **Image Preprocessing**: Convert the input image to the LAB color space
2. **Model Inference**: Use a pre-trained deep learning model to predict the color channels
3. **Post-processing**: Combine the predicted colors with the original luminance channel
4. **Color Space Conversion**: Convert back to RGB for display

## 🖥️ Running the Application

### Prerequisites
- Python 3.7+
- Required packages (install via `pip install -r requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "Colorize Black and White Images"
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Streamlit App

1. Navigate to the project directory
2. Run the Streamlit app:
   ```bash
   streamlit run streamlit/app.py
   ```
3. Open your web browser and go to `http://localhost:8501`
4. Upload a black and white image and see it come to life!

## 📦 Requirements

Create a `requirements.txt` file with the following content:

```
streamlit==1.29.0
opencv-python-headless==4.8.1.78
numpy==1.24.3
Pillow==10.0.0
```

## 🎯 Features

- Simple and intuitive web interface
- Fast colorization using pre-trained models
- Download colorized images
- Responsive design that works on different screen sizes

## 📝 Notes

- For best results, use high-quality black and white photos with good contrast
- The model works best with human faces, landscapes, and common objects
- Colorization quality may vary depending on the input image

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- The pre-trained model is based on the work of Richard Zhang et al. (2016) - [Colorful Image Colorization](https://richzhang.github.io/colorization/)
- Thanks to the OpenCV and Streamlit communities for their amazing tools

---
