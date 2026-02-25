# Text Detection Using EasyOCR

## Overview
This project demonstrates real-time text detection from video streams using the EasyOCR library and OpenCV. The application captures video from a webcam, processes frames to detect and extract text, and displays the results with bounding boxes and labels overlaid on the video feed.

## Features
- Real-time text detection from webcam video
- Uses EasyOCR for robust multilingual text recognition
- Optimized for speed by processing every Nth frame and resizing frames
- Visualizes detected text with bounding boxes and labels
- Simple and easy-to-understand code structure

## Requirements
- Python 3.6+
- OpenCV (`cv2`)
- EasyOCR

## Installation
1. Clone this repository or download the project files.
2. Install the required Python packages:
   ```powershell
   pip install opencv-python easyocr
   ```

## Usage
### Video Text Detection
Run the following command to start real-time text detection from your webcam:
```powershell
python VideoTextDetection.py
```
- The application will open a window displaying the webcam feed.
- Detected text will be highlighted with green bounding boxes and labeled.
- Press `q` to quit the application.

### Image Text Detection
If you have an `ImageTextDetection.py` script, you can use it to detect text in static images. (Refer to the script for usage details.)

## How It Works
- The script captures video frames from the default webcam.
- To improve performance, it processes every 10th frame (configurable) and resizes frames to half their original size before running OCR.
- Detected text regions are scaled back to the original frame size for accurate visualization.
- Results are cached and displayed on skipped frames to maintain a smooth video experience.

## Customization
- **Frame Skipping:** Adjust the `skip_frames` variable to change how often OCR is performed.
- **Frame Resize:** Modify the resize factor in the code for different speed/accuracy trade-offs.
- **Language Support:** Change the language list in `easyocr.Reader(['en'], gpu=False)` to support other languages (see EasyOCR documentation).

## Troubleshooting
- If the webcam does not open, ensure it is connected and not used by another application.
- For GPU acceleration, set `gpu=True` in the EasyOCR reader if you have a compatible GPU and CUDA installed.
- If you encounter errors related to missing Win32 UI support, ensure your OpenCV installation includes GUI support.

## References
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Documentation](https://docs.opencv.org/)
