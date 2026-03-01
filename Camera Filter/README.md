# Camera Filter

A real-time camera filter application built with OpenCV and Python that allows you to apply various computer vision filters to your live webcam feed.

## Features

- **Real-time Processing**: Apply filters to your webcam feed with minimal lag
- **9 Different Filters**: Switch between multiple image processing filters on-the-fly
- **Performance Optimized**: Configured for smooth 1080p @ 30 FPS performance
- **Frame Capture**: Save filtered frames with a single keypress
- **FPS Monitoring**: Real-time frame rate display

## Available Filters

1. **BGR (Original)** - Raw camera feed
2. **Canny Edge Detection** - Detects edges in the image
3. **Grayscale** - Converts to black and white
4. **Gaussian Blur** - Applies a smoothing blur effect
5. **Laplacian Edge Detection** - Alternative edge detection method
6. **Sobel X** - Detects vertical edges
7. **Sobel Y** - Detects horizontal edges
8. **HSV Format** - Displays image in Hue-Saturation-Value color space
9. **Threshold Binary** - Creates a binary black/white image

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- A working webcam/camera

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install opencv-python numpy
```

## Usage

1. Run the application:
```bash
python CameraFilter.py
```

2. Use keyboard controls to interact with the application:
   - **Keys 1-9**: Switch between different filters
   - **S**: Save the current frame as a JPEG image
   - **Q**: Quit the application

3. The application will display:
   - Live filtered video feed in a window titled "Camera Filter"
   - FPS updates in the console every 30 frames
   - Current filter name when switching filters

## Controls

| Key | Action |
|-----|--------|
| `1` | BGR (Original) |
| `2` | Canny Edge Detection |
| `3` | Grayscale |
| `4` | Gaussian Blur |
| `5` | Laplacian Edge Detection |
| `6` | Sobel X |
| `7` | Sobel Y |
| `8` | HSV Format |
| `9` | Threshold Binary |
| `S` | Save current frame |
| `Q` | Exit application |

## Saved Images

When you press the `S` key, the current filtered frame is saved as a JPEG file with a timestamp filename format:
```
YYYYMMDD-HHMMSS.jpg
```

Example: `20260302-143025.jpg`

## Performance

The application is configured for optimal performance:
- Resolution: 1920x1080 (Full HD)
- Target FPS: 30
- Minimal buffer size for reduced lag
- Efficient filter switching without restart

## Troubleshooting

### Camera Not Opening
If you see "Error: Could not open camera", try:
- Ensure your camera is properly connected
- Check if another application is using the camera
- Verify camera permissions on your system
- Try changing the camera index from `0` to `1` or `2` in the code

### Low FPS
If you experience low frame rates:
- Reduce the resolution by modifying `CAP_PROP_FRAME_WIDTH` and `CAP_PROP_FRAME_HEIGHT`
- Close other resource-intensive applications
- Ensure your camera supports the configured resolution and FPS

### Empty Frames
If you receive "Warning: Empty frame received":
- Check your camera connection
- Restart the application
- Verify camera drivers are up to date

## Technical Details

### Filter Implementations

- **Canny**: Uses Gaussian blur preprocessing with thresholds (30, 50)
- **Gaussian Blur**: 15x15 kernel size
- **Sobel**: 5x5 kernel size for gradient detection
- **Threshold**: Binary threshold at 127

### Camera Settings

The application configures the camera with:
```python
video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
video.set(cv2.CAP_PROP_FPS, 30)
```

## License

This project is open source and available for educational and personal use.

## Contributing

Feel free to fork this project and add your own filters or improvements!

## Future Enhancements

Potential features to add:
- More advanced filters (Sepia, Cartoon effect, etc.)
- Filter parameter adjustment with sliders
- Video recording capability
- Face detection and tracking
- Multi-camera support
- Custom filter presets

## Author

Created as a Computer Vision project demonstrating real-time image processing with OpenCV.