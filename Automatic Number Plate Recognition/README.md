# Automatic Number Plate Recognition (ANPR)

This project is an end-to-end Automatic Number Plate Recognition (ANPR) system that leverages deep learning, object detection, OCR, and data processing to detect vehicles and license plates in video streams, extract plate numbers, and visualize results.

## Features
- **YOLO-based License Plate Detection**: Custom-trained YOLO model for accurate license plate localization.
- **Vehicle Detection**: Integrates a vehicle detection model for multi-object tracking.
- **SORA Integration**: Advanced object tracking and frame management.
- **EasyOCR**: Extracts license plate text from detected regions.
- **Data Export**: Detection results are saved frame-by-frame to CSV.
- **Data Cleaning**: Handles missing values and removes duplicates for reliable analytics.
- **Visualization**: Visualizes detections and vehicle tracking on video.

## Project Structure
```
app/
    app.py                  # Streamlit app entry point
    video_plate_detection.py# Detection pipeline
    utils.py                # Utility functions
    ...
data/
    input/                  # Input videos/images
    output/                 # Output CSVs and results
models/                     # Trained models
Training a Licence Plate Detector Model/
    ...                     # Training scripts and datasets
```

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download or train YOLO and vehicle detection models**
4. **Run the Streamlit app**
   ```bash
   streamlit run app/app.py
   ```

## Steps Overview
1. Train YOLO on a license plate dataset.
2. Export the trained model and integrate with vehicle detection.
3. Use SORA for tracking and EasyOCR for plate text extraction.
4. Export detection results to CSV.
5. Clean and process the CSV for visualization.
6. Visualize results in the Streamlit app.

## Credits
- YOLO for object detection
- SORA for tracking
- EasyOCR for text extraction

## License
MIT License
