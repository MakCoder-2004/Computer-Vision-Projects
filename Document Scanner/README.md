# Document Scanner

A real-time document scanner application using OpenCV and Python that detects documents from a webcam feed, performs perspective transformation, and outputs a clean scanned image.

## Features

- **Real-time Document Detection**: Automatically detects rectangular documents in the webcam feed
- **Edge Detection**: Uses Canny edge detection and contour analysis to find document boundaries
- **Perspective Transformation**: Warps the detected document to a flat, top-down view
- **Live Preview**: Shows both the original feed with detected contours and the warped output
- **Image Saving**: Save scanned documents by pressing the 's' key

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

## Installation

1. Clone or download this repository

2. Install the required dependencies:
```bash
pip install opencv-python numpy
```

3. Ensure you have a working webcam connected to your computer

## Usage

1. Run the application:
```bash
python DocumentScanner.py
```

2. Position a document in front of your webcam. The document should have clear edges and be well-lit.

3. The application will display two windows:
   - **WorkFlow**: Shows the original view with detected contours and the warped document side-by-side
   - **ImageWarped**: Shows only the scanned document (appears when a document is detected)

4. **Controls**:
   - Press **'s'** to save the scanned document to `Scanned/myImage.jpg`
   - Press **'q'** to quit the application

## How It Works

1. **Image Preprocessing**:
   - Converts the frame to grayscale
   - Applies Gaussian blur to reduce noise
   - Uses Canny edge detection to find edges
   - Dilates and erodes to enhance contours

2. **Contour Detection**:
   - Finds all external contours in the processed image
   - Filters contours by area (minimum 5000 pixels)
   - Identifies quadrilateral shapes (4-sided polygons)
   - Selects the largest quadrilateral as the document

3. **Perspective Transformation**:
   - Reorders corner points to a standard format
   - Applies perspective transform to get a flat view
   - Crops and resizes to the specified dimensions (540x640)

## Configuration

You can modify the following parameters in `DocumentScanner.py`:

- **Image dimensions**:
  ```python
  widthImg = 540
  heightImg = 640
  ```

- **Webcam settings**:
  ```python
  cap = cv2.VideoCapture(0)  # Change 0 to use a different camera
  cap.set(10, 150)           # Brightness setting
  ```

- **Edge detection sensitivity**:
  ```python
  imgCanny = cv2.Canny(imgBlur, 200, 200)  # Adjust threshold values
  ```

- **Minimum contour area**:
  ```python
  if area > 5000:  # Adjust this value to change sensitivity
  ```

## Troubleshooting

### Camera not detected
- Ensure your webcam is properly connected
- Try changing the camera index from `0` to `1` or `2` in `cap = cv2.VideoCapture(0)`
- Check if other applications are using the webcam

### Document not being detected
- Ensure the document has clear, straight edges
- Improve lighting conditions
- Place the document on a contrasting background
- Adjust the minimum area threshold in `GetContours()` function
- Try adjusting the Canny edge detection thresholds

### Poor scan quality
- Ensure the document is well-lit and flat
- Adjust the brightness setting: `cap.set(10, value)` (try values between 100-200)
- Modify the Gaussian blur or Canny thresholds for better edge detection

## Output

Scanned documents are saved in the `Scanned/` directory as `myImage.jpg`. Make sure this directory exists or create it manually before saving.

## Project Structure

```
Document Scanner/
├── DocumentScanner.py    # Main application file
├── Scanned/             # Directory for saved scanned images
│   └── myImage.jpg      # Saved scans (created when you press 's')
└── README.md            # This file
```

## Technical Details

- **Language**: Python
- **Libraries**: OpenCV 4.x, NumPy
- **Image Processing Techniques**: Grayscale conversion, Gaussian blur, Canny edge detection, morphological operations (dilation/erosion), contour detection, perspective transformation

## Future Enhancements

- Add support for multiple document formats (A4, Letter, etc.)
- Automatic image enhancement (brightness, contrast, sharpness)
- Batch scanning mode
- OCR integration for text extraction
- GUI for easier parameter adjustment
- Support for scanning from image files

## License

This project is open-source and available for educational and personal use.

## Author

Created as part of a Computer Vision project collection.

---

**Note**: This application works best with documents that have clear, distinct edges and are placed on a contrasting background. Ensure good lighting for optimal results.

