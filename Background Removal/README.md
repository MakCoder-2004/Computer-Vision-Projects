# Background Removal

This project uses MediaPipe and OpenCV to perform real-time background removal and replacement on images and webcam video streams. It leverages a pre-trained TensorFlow Lite segmentation model to separate the foreground (person) from the background, allowing for effects such as blurring, solid color backgrounds, or custom image backgrounds.

## Features
- Real-time webcam background removal
- Custom background image support
- Blurred or solid color background options
- Save processed frames
- Easy-to-use Python scripts

## Requirements
- Python 3.7+
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

## Setup
1. **Clone the repository:**
   ```powershell
   git clone <your-repo-url>
   cd "Background Removal"
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Download the segmentation model:**
   - Place `selfie_segmenter.tflite` in the project root (already included).

## Usage
### Webcam Background Removal
Run the following command to start real-time background removal using your webcam:
```powershell
python VideoBackgroundRemoval.py
```

- Press `q` to quit.
- Press `s` to save the current frame.
- Press `r` to toggle between blur/solid and custom image background modes.

### Image Background Removal
You can use `ImageBackgroundRemoval.py` to process static images. (See script for details.)

## File Structure
- `VideoBackgroundRemoval.py` — Main script for webcam background removal
- `ImageBackgroundRemoval.py` — Script for image background removal
- `model/selfie_segmenter.tflite` — Pre-trained segmentation model
- `images/background.jpg` — Example custom background image
- `captures/` — Saved output frames

## License
This project is licensed under the MIT License.
