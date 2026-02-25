import cv2
import numpy as np
from PIL import Image

#--------------------------------
# RGB to BGR conversion function
#--------------------------------
def rgb_to_bgr(rgb):
    r, g, b = rgb
    return b, g, r

#---------------------------
# Get the HSV color limits
#---------------------------
def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

#----------------------------------------------------
# Detect Function to capture video and detect color
#----------------------------------------------------
def detect(color, capture):

    while True:
        success, frame = video.read()
        if not success:
            break

        # Optional: Blur the frame to reduce noise
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)

        # Convert the captured frame from BGR to HSV color space
        hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Get the lower and upper HSV limits for the specified BGR color (red in this case)
        lower_limit, upper_limit = get_limits(color)

        # Create a binary mask where the detected color is white and the rest is black
        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)

        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding box around the largest contour if area is above threshold
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > 500:  # Minimum area threshold to avoid noise
                x, y, w, h = cv2.boundingRect(largest_contour)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Color Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#--------
# Main
#--------
if __name__ == "__main__":
    print("++++++++++++++++++++++++++++++++++++++++")
    print("    Color Detection From Your Camera    ")
    print("++++++++++++++++++++++++++++++++++++++++"+"\n")

    print("Write the RGB color value you want to detect (e.g., 255,0,0 for red):")
    color_input = input()
    try:
        color = tuple(map(int, color_input.split(',')))
        if len(color) != 3 or not all(0 <= c <= 255 for c in color):
            raise ValueError
    except ValueError:
        print("Invalid input. Please enter RGB values in the format R,G,B (e.g., 255,0,0).")
        exit(1)
    b, g, r = rgb_to_bgr(color)
    color = [b, g, r]


    # -----------------------------------------
    # Start video capture and color detection
    # -----------------------------------------
    video = cv2.VideoCapture(0)
    print("Starting your Camera...")
    if not video.isOpened():
        print("Error: Could not open video stream.")
        exit(1)
    print("Camera started successfully. Press 'q' to quit.")
    print("Starting color detection...")
    detect(color, video)

    video.release()
    cv2.destroyAllWindows()