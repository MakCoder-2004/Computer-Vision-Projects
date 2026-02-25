# Color Detection

This project is a simple real-time color detection tool using OpenCV and Python. It allows users to detect a specific color from their webcam feed by specifying the RGB value of the color they want to track. The program highlights the detected color region in the video stream with a bounding box.

## Features
- Real-time color detection using your computer's camera
- User input for any RGB color to detect
- HSV color space conversion for robust color detection
- Noise reduction using Gaussian blur and morphological operations
- Bounding box drawn around the largest detected color region
- Easy to use: just run the script and enter the desired color

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pillow (PIL)

Install dependencies with:
```bash
pip install opencv-python numpy pillow
```

## Usage
1. Run the script:
   ```bash
   python ColorCounter.py
   ```
2. Enter the RGB value of the color you want to detect (e.g., `255,0,0` for red).
3. The camera window will open. The program will highlight the detected color region with a green bounding box.
4. Press `q` to quit the application.

## How It Works
- The script captures video from your webcam.
- It converts each frame to the HSV color space for better color segmentation.
- The user-specified RGB color is converted to BGR (OpenCV format), then to HSV to determine the detection range.
- A mask is created to isolate the specified color.
- Morphological operations clean up noise in the mask.
- The largest detected region is highlighted with a bounding box.

## File Structure
- `ColorCounter.py`: Main script for color detection.

## Notes
- The detection threshold and area can be adjusted in the script for different use cases.
- The script currently detects only one color at a time.