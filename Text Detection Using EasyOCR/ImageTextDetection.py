# ---------------------------------------
# Loading Libraries
# ---------------------------------------
import easyocr
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------
# Loading Image
# ---------------------------------------
image_path = "test-en.jpg"
image = cv2.imread(image_path)

# ---------------------------------------
# Detecting Text
# ---------------------------------------
reader = easyocr.Reader(['en'])
text = reader.readtext(image)

# ---------------------------------------
# Displaying Detected Text
# ---------------------------------------
for t in text:
    bbox, text, confidence = t
    print(f"Detected Text: {text}, Confidence: {confidence}")
    top_left = tuple(map(int, bbox[0]))
    bottom_right = tuple(map(int, bbox[2]))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 5)

    # Calculate text size
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.2
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    # Add padding
    pad_x, pad_y = 10, 10
    rect_top_left = (top_left[0], top_left[1] - text_height - pad_y)
    rect_bottom_right = (top_left[0] + text_width + pad_x, top_left[1])

    # Draw filled rectangle for text background
    cv2.rectangle(image, rect_top_left, rect_bottom_right, (0, 255, 0), -1)

    # Draw text
    text_org = (top_left[0] + 5, top_left[1] - 5)
    cv2.putText(image, text, text_org, font, font_scale, (0,0,0), thickness, cv2.LINE_AA)

# Display using OpenCV instead of matplotlib
cv2.imshow("Text Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
