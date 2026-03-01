import cv2
import time
import numpy as np

# ---------------------------------------------
# Set camera properties for better performance
# ---------------------------------------------
video = cv2.VideoCapture(0)

# Check if camera opened successfully
if not video.isOpened():
    print("Error: Could not open camera. Please check if your camera is connected and available.")
    exit()

video.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce lag
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
video.set(cv2.CAP_PROP_FPS, 30)


frame_count = 0
start_time = time.time()
current_filter = 1  # Default to BGR format

# ---------------------------------------------
# Filter names for reference
# ---------------------------------------------
filter_names = {
    1: "BGR (Original)",
    2: "Canny Edge Detection",
    3: "Grayscale",
    4: "Gaussian Blur",
    5: "Laplacian Edge Detection",
    6: "Sobel X",
    7: "Sobel Y",
    8: "HSV Format",
    9: "Threshold Binary"
}

print("Camera Filter Controls:")
print("1-9: Change filter (1=BGR, 2=Canny, 3=Grayscale, 4=Blur, 5=Laplacian, 6=SobelX, 7=SobelY, 8=HSV, 9=Threshold)")
print("S: Save current frame")
print("Q: Exit")
print()

# ---------------------------------------------
# Function to apply selected filter
# ---------------------------------------------
def apply_filter(frame, filter_num):
    """Apply selected filter to the frame"""
    if filter_num == 1:
        # BGR format (original)
        return frame
    elif filter_num == 2:
        # Canny edge detection
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        return cv2.Canny(frame_blur, 30, 50)
    elif filter_num == 3:
        # Grayscale
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_num == 4:
        # Gaussian Blur
        return cv2.GaussianBlur(frame, (15, 15), 0)
    elif filter_num == 5:
        # Laplacian edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F)
    elif filter_num == 6:
        # Sobel X
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    elif filter_num == 7:
        # Sobel Y
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    elif filter_num == 8:
        # HSV format
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    elif filter_num == 9:
        # Threshold binary
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return thresh
    else:
        return frame


# ---------------------------------------------
# Main loop to read frames and apply filters
# ---------------------------------------------
while True:
    success, frame = video.read()

    # Check if frame was successfully read
    if not success or frame is None or frame.size == 0:
        print("Warning: Empty frame received, skipping...")
        exit()

    frame_count += 1

    # Apply selected filter
    display_frame = apply_filter(frame, current_filter)

    # Display the frame in a single window
    cv2.imshow('Camera Filter', display_frame)

    # ---------------------------------------------
    # Show FPS every 30 frames
    # ---------------------------------------------
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"FPS: {fps:.1f} | Current Filter: {filter_names[current_filter]}")

    # Single key check (much faster than multiple calls)
    key = cv2.waitKey(1) & 0xFF

    # ---------------------------------------------
    # Change filter (1-9)
    # ---------------------------------------------
    if ord('1') <= key <= ord('9'):
        current_filter = key - ord('0')
        print(f"Switched to Filter {current_filter}: {filter_names[current_filter]}")

    # ---------------------------------------------
    # Save the image when 's' key is pressed
    # Save the image when 's' key is pressed
    elif key == ord('s'):
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f'{time_stamp}.jpg', display_frame)
        print(f"Image saved: {time_stamp}.jpg with {filter_names[current_filter]} filter")

    # ---------------------------------------------
    # Exit the loop when 'q' key is pressed
    # ---------------------------------------------
    elif key == ord('q'):
        print("Exiting...")
        break

video.release()
cv2.destroyAllWindows()
print(f"Final FPS: {frame_count / (time.time() - start_time):.1f}")