# Parking Spots Detection using Deep Learning

A real-time parking spot availability detection system using deep learning and computer vision. This project uses a pre-trained ResNet50 model fine-tuned to classify individual parking spots as occupied or empty.

## 🎯 Project Overview

This project implements an intelligent parking management system that:
- **Detects parking spot boundaries** using image processing on a mask
- **Classifies spots in real-time** as occupied or empty using a trained CNN model
- **Processes video streams** to identify available parking spots
- **Displays live statistics** of available spots during video processing

## 📊 Project Features

- ✅ **AI-Powered Classification**: Uses ResNet50 deep learning model for accurate car detection
- ✅ **Real-time Processing**: Processes video frames with GPU optimization for fast inference
- ✅ **Motion Detection**: Implements motion detection to optimize inference on changed areas
- ✅ **Batch Processing**: Processes multiple parking spots in batches for efficiency
- ✅ **GPU Acceleration**: Utilizes TensorFlow GPU optimization and mixed precision (FP16)
- ✅ **Video Export**: Outputs annotated video with parking spot status visualization

## 📁 Project Structure

```
Parking Spots Detection/
├── data/
│   ├── inputs/                          # Input videos
│   │   └── parking_lot_video.mp4
│   ├── outputs/                         # Output videos with detections
│   │   └── parking_lot_output.mp4
│   └── model_data/                      # Parking spot mask
│       └── mask.png
├── Car Occurrence Model/
│   ├── Car Occurrence Model.ipynb       # Training notebook
│   ├── model/
│   │   └── car_occurrence_model.keras   # Trained model
│   └── data/
│       ├── train/
│       │   ├── empty/                   # Training images of empty spots
│       │   └── not_empty/               # Training images of occupied spots
│       └── test/
│           ├── empty/                   # Test images of empty spots
│           └── not_empty/               # Test images of occupied spots
├── ParkingSlotDetector.py               # Main detection script
├── README.md                            # This file
├── .gitignore                           # Git ignore rules
└── plan.txt                             # Project development plan
```

## 🛠️ Technologies Used

- **Python 3.x**
- **OpenCV** - Image and video processing
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computations
- **ResNet50** - Pre-trained CNN architecture

## 📋 Prerequisites

- Python 3.7+
- NVIDIA GPU (recommended for real-time processing)
- CUDA Toolkit (if using GPU)
- cuDNN (if using GPU)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Parking-Spots-Detection.git
   cd Parking-Spots-Detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **GPU Setup (Optional but recommended)**
   - Install CUDA Toolkit and cuDNN
   - Verify GPU support: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

## 🚀 Usage

Run the main detection script:

```bash
python ParkingSlotDetector.py
```

**Script Output:**
- Displays real-time video with parking spot annotations
- Shows available spots count
- Saves annotated video to `data/outputs/parking_lot_output.mp4`
- Press 'q' to quit

### How It Works

1. **Spot Detection**: Reads `mask.png` and uses connected component analysis to identify individual parking spots
2. **Motion Detection**: Detects areas with motion to optimize inference
3. **AI Inference**: Feeds cropped parking spot images to the trained model
4. **Classification**: Determines if each spot is empty (green) or occupied (red)
5. **Visualization**: Draws rectangles and displays statistics on the video

## 🧠 Model Training

The car occurrence model was trained using:
- **Architecture**: ResNet50 (pre-trained on ImageNet)
- **Dataset**: Custom dataset of parking spot images
- **Classes**: 2 (empty, occupied)
- **Training**: See `Car Occurrence Model/Car Occurrence Model.ipynb` for details

To retrain the model:
```bash
cd "Car Occurrence Model"
jupyter notebook "Car Occurrence Model.ipynb"
```

## ⚙️ Configuration

Key parameters in `ParkingSlotDetector.py`:
- `step = 45` - Process every 45th frame for efficiency
- `batch_size = 16` - Batch size for model inference
- `threshold = 0.5` - Classification threshold
- `scale_factor = 0.75` - Display resolution scaling

## 📊 Performance

- **GPU Optimization**: Mixed precision (FP16) for faster inference
- **XLA Acceleration**: Enabled for TensorFlow optimization
- **Batch Processing**: Processes up to 16 spots simultaneously
- **Motion Detection**: Skips unchanged areas to save compute

## 🔧 Troubleshooting

**No GPU detected:**
- Verify CUDA installation: `nvidia-smi`
- Check TensorFlow GPU support: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

**Slow processing:**
- Enable GPU memory growth in the script (already implemented)
- Reduce video resolution
- Increase `step` parameter to skip more frames

**Model not found:**
- Ensure `Car Occurrence Model/model/car_occurrence_model.keras` exists
- Check relative paths if running from different directory

## 📈 Development Plan

- [x] Gathering training data for the model
- [x] Creating training notebook using ResNet50
- [x] Ensuring model accuracy
- [x] Exporting and integrating the model
- [x] Creating real-time detection script
- [x] Testing in real-time scenarios
- [x] Performance monitoring and optimization
- [x] Documentation and user guide
- [x] Sharing on GitHub

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Share feedback

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📧 Contact

For questions or suggestions, feel free to reach out!

---

**Note**: This project demonstrates practical applications of deep learning in computer vision, specifically for smart parking management systems.

