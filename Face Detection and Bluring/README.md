# Face Detection and Blurring

This project demonstrates real-time face detection and blurring in images and video streams using MediaPipe and OpenCV. It is designed for privacy protection and anonymization in computer vision applications.

## Features
- Real-time face detection in video streams (webcam)
- Face blurring for privacy
- Image face detection and blurring (see `ImageFaceDetectionAndBluring.py`)
- Uses MediaPipe's Face Detection model (`detector.tflite`)
- Easy to run and extend

## Requirements
- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe

## Installation
1. Clone the repository:
   ```sh
   git clone <your-repo-url>
   cd Face Detection and Bluring
   ```
2. Install dependencies:
   ```sh
   pip install opencv-python mediapipe
   ```

## Usage
### Video Face Detection and Blurring
Run the following command to start the webcam face detection and blurring:
```sh
python VideoFaceDetectionAndBluring.py
```
Press `q` to quit the video window.

### Image Face Detection and Blurring
Run the following command to process a static image:
```sh
python ImageFaceDetectionAndBluring.py
```

## Files
- `VideoFaceDetectionAndBluring.py`: Real-time face detection and blurring from webcam
- `ImageFaceDetectionAndBluring.py`: Face detection and blurring for static images
- `detector.tflite`: MediaPipe face detection model
- `test.jpg`: Sample image for testing

## Notes
- Ensure your webcam is connected for video processing.
- The face detection model (`detector.tflite`) is required for both scripts.
- Adjust the blurring kernel size in the code if needed for different levels of blurring.