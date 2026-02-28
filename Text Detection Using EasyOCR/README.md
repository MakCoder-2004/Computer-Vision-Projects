# Text Detection Using EasyOCR

## Overview
This project demonstrates real-time text detection from video streams using the EasyOCR library, OpenCV, and a Tkinter GUI. The application captures video from a webcam, processes frames to detect and extract text, and displays the results with bounding boxes and labels overlaid on the video feed in a user-friendly window.

## Features
- Real-time text detection from webcam video
- Uses EasyOCR for robust multilingual text recognition (GPU acceleration enabled by default)
- Tkinter GUI with Start/Stop buttons for easy control
- Optimized for speed: frame capture and OCR run in separate threads to minimize lag and frame drops
- Visualizes detected text with bounding boxes and labels
- Simple and easy-to-understand code structure

## Requirements
- Python 3.6+
- OpenCV (`cv2`)
- EasyOCR
- Tkinter (included with standard Python)

## Installation
1. Clone this repository or download the project files.
2. Install the required Python packages:
   ```powershell
   pip install opencv-python easyocr pillow
   ```

## Usage
### Video Text Detection
Run the following command to start real-time text detection from your webcam:
```powershell
python VideoTextDetection.py
```
- The application will open a Tkinter window displaying the webcam feed.
- Use the **Start Detection** button to begin text detection.
- Use the **Stop Detection** button to stop the video and release the webcam.
- Detected text will be highlighted with green bounding boxes and labeled.
- Close the window to exit the application.

### Image Text Detection
If you have an `ImageTextDetection.py` script, you can use it to detect text in static images. (Refer to the script for usage details.)

## How It Works
- The script captures video frames from the default webcam.
- Frame capture and OCR run in separate threads for smooth performance.
- To improve speed, frames are resized before OCR and only the latest frame is processed.
- Detected text regions are scaled back to the original frame size for accurate visualization.
- Results are cached and displayed on skipped frames to maintain a smooth video experience.

## Customization
- **Frame Skipping:** Adjust the `skip_frames` variable to change how often OCR is performed.
- **Frame Resize:** Modify the resize factor in the code for different speed/accuracy trade-offs.
- **Language Support:** Change the language list in `easyocr.Reader(['en'], gpu=True)` to support other languages (see EasyOCR documentation).

## Troubleshooting
- If the webcam does not open, ensure it is connected and not used by another application.
- For GPU acceleration, ensure you have a compatible GPU and CUDA installed. The application uses GPU by default.
- If you encounter errors related to missing Win32 UI support, ensure your OpenCV installation includes GUI support.

## References
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Documentation](https://docs.opencv.org/)
