# Face Mask Detection using YOLO

A real-time face mask detection system built with YOLOv8 (Ultralytics) that can detect whether people are wearing masks correctly, incorrectly, or not at all. The system provides simplified binary classification: **mask** or **no_mask**.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-red.svg)

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Training](#model-training)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [License](#license)

## ✨ Features

- **Real-time detection** using webcam feed
- **Simplified classification**: 
  - `mask` - Person wearing a mask correctly
  - `no_mask` - Person not wearing a mask or wearing it incorrectly
- **Color-coded bounding boxes**:
  - 🟢 Green for mask detected
  - 🔴 Red for no mask
- **Confidence scores** displayed on each detection
- **ONNX model support** for faster inference
- **Custom YOLO training** pipeline from XML annotations

## 📁 Project Structure

```
Face Mask Detection/
│
├── FaceMaskDetector.py          # Main detection script for real-time webcam
├── Face Mask Detection Model using YOLO.ipynb  # Jupyter notebook for model training
├── config.yaml                   # YOLO dataset configuration
├── classes.txt                   # Class names file (3 original classes)
├── archive.zip                   # Original dataset archive
│
├── data/                         # Raw dataset
│   ├── annotations/              # Pascal VOC XML annotation files
│   │   ├── maksssksksss0.xml
│   │   ├── maksssksksss1.xml
│   │   └── ...
│   ├── images/                   # Original images
│   └── labels/                   # Converted YOLO format labels
│
├── dataset/                      # Organized dataset for training
│   ├── train/                    # Training set (80%)
│   │   ├── images/
│   │   └── labels/
│   └── val/                      # Validation set (20%)
│       ├── images/
│       └── labels/
│
├── models/                       # Trained models
│   ├── best.pt                   # Best model checkpoint (PyTorch)
│   └── best.onnx                 # Exported ONNX model for inference
│
└── utils/                        # Utility scripts
    ├── xml_to_yolo.py            # Convert XML annotations to YOLO format
    └── prepare_dataset.py        # Split dataset into train/val sets
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time detection)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/face-mask-detection.git
   cd face-mask-detection
   ```

2. **Install required packages**
   ```bash
   pip install ultralytics opencv-python numpy
   ```

3. **Verify installation**
   ```bash
   python -c "import cv2, ultralytics; print('All packages installed successfully!')"
   ```

## 📊 Dataset

The project uses a face mask detection dataset with three original classes:
- **Class 0**: `mask_weared_incorrect` - Mask worn incorrectly
- **Class 1**: `with_mask` - Mask worn correctly
- **Class 2**: `without_mask` - No mask

### Simplified Classification
For practical use, the model output is mapped to binary classification:
- **"mask"** ← Class 1 (with_mask)
- **"no_mask"** ← Class 0 (mask_weared_incorrect) + Class 2 (without_mask)

### Dataset Preparation

The dataset goes through the following pipeline:

1. **Convert XML to YOLO format**
   ```bash
   cd utils
   python xml_to_yolo.py
   ```
   - Converts Pascal VOC XML annotations to YOLO format
   - Automatically detects all classes from XML files
   - Creates `classes.txt` file
   - Outputs normalized bounding boxes

2. **Split into Train/Val sets**
   ```bash
   python prepare_dataset.py
   ```
   - Splits data into 80% training, 20% validation
   - Organizes images and labels into proper YOLO structure
   - Uses random seed (42) for reproducibility

## 🚀 Usage

### Real-time Face Mask Detection

Run the main detection script:
```bash
python FaceMaskDetector.py
```

**Controls:**
- Press `q` to quit the application

The system will:
1. Open your webcam
2. Detect faces in real-time
3. Classify each face as "mask" or "no_mask"
4. Display bounding boxes with labels and confidence scores

### Using the Jupyter Notebook

Open and run the training notebook:
```bash
jupyter notebook "Face Mask Detection Model using YOLO.ipynb"
```

The notebook contains:
- Data exploration and visualization
- Model training configuration
- Training process with metrics
- Model evaluation
- Export to ONNX format

## 🎯 Model Training

### Training Configuration

The model is trained using YOLOv8 with the following configuration (in `config.yaml`):

```yaml
path: dataset/
train: train/images
val: val/images
nc: 3  # Number of classes
names:
  0: mask_weared_incorrect
  1: with_mask
  2: without_mask
```

### Training Process

1. **Prepare the dataset** (see Dataset Preparation above)

2. **Train the model**
   - Open the Jupyter notebook or create a training script
   - Load YOLOv8 base model
   - Train with your configuration:
   ```python
   from ultralytics import YOLO
   
   model = YOLO('yolov8n.pt')  # Load pretrained model
   results = model.train(
       data='config.yaml',
       epochs=100,
       imgsz=640,
       batch=16
   )
   ```

3. **Export to ONNX** (for optimized inference)
   ```python
   model = YOLO('models/best.pt')
   model.export(format='onnx')
   ```

## 🔍 How It Works

### Detection Pipeline

1. **Frame Capture**: Captures video frames from webcam using OpenCV
2. **Inference**: Passes each frame to YOLO model for detection
3. **Post-processing**: 
   - Extracts bounding boxes, class IDs, and confidence scores
   - Maps class IDs to simplified labels using `label_map`
4. **Visualization**: Draws bounding boxes and labels on the frame
5. **Display**: Shows annotated frame in real-time

### Label Mapping Logic

```python
label_map = {
    0: "no_mask",      # mask_weared_incorrect → no_mask
    1: "mask",         # with_mask → mask
    2: "no_mask"       # without_mask → no_mask
}
```

### Color Coding

```python
color = (0, 255, 0) if label == "mask" else (0, 0, 255)
# Green (BGR) for mask, Red (BGR) for no_mask
```

## 📦 File Descriptions

### Main Files

- **`FaceMaskDetector.py`**: Real-time detection application with webcam integration. Loads the ONNX model and performs inference on video frames with custom visualization.

- **`Face Mask Detection Model using YOLO.ipynb`**: Jupyter notebook documenting the complete training workflow including data loading, model configuration, training, evaluation, and export.

- **`config.yaml`**: YOLO dataset configuration file specifying dataset paths, number of classes, and class names.

- **`classes.txt`**: Text file listing all class names (one per line) in order.

### Utility Scripts

- **`utils/xml_to_yolo.py`**: Converts Pascal VOC XML annotations to YOLO format (class_id, x_center, y_center, width, height). Includes functions to:
  - Parse XML files
  - Convert bounding box coordinates
  - Generate class mapping automatically
  - Create classes.txt file

- **`utils/prepare_dataset.py`**: Organizes dataset into YOLO-compatible structure by:
  - Splitting data into train (80%) and validation (20%) sets
  - Copying images and labels to appropriate directories
  - Validating that each image has a corresponding label file

### Model Files

- **`models/best.pt`**: Best model checkpoint saved during training in PyTorch format. Used for further training or fine-tuning.

- **`models/best.onnx`**: Exported ONNX model optimized for inference. Used by the detection script for faster performance.

### Data Directories

- **`data/annotations/`**: Contains Pascal VOC XML annotation files from the original dataset.

- **`data/images/`**: Original dataset images.

- **`data/labels/`**: YOLO format label files (converted from XML).

- **`dataset/train/`**: Training images and labels (80% of data).

- **`dataset/val/`**: Validation images and labels (20% of data).

## 📋 Requirements

```
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.19.0
PyYAML>=5.4.0
```

### Optional (for training)
```
jupyter
matplotlib
pandas
```

## 🎓 Model Performance

- **Input Size**: 640x640 pixels
- **Architecture**: YOLOv8n (nano) - optimized for speed
- **Inference Time**: ~10-30ms per frame (depends on hardware)
- **Classes**: 3 original classes, mapped to 2 simplified categories

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the excellent YOLO implementation
- Dataset creators for providing annotated face mask images
- OpenCV community for computer vision tools

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: Make sure you have a working webcam connected before running the detection script. The model files (`best.pt` or `best.onnx`) must be present in the `models/` directory.

