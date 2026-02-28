import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------
# Load Image
# ------------------
image_path = "test.jpg"
img = cv2.imread(image_path)
ih, iw, _ = img.shape

# Convert to RGB
rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert image to MediaPipe format
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

# ------------------
# Create Face Detector
# ------------------
base_options = python.BaseOptions(model_asset_path='model/detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# ------------------
# Detect Faces
# ------------------
detection_result = detector.detect(mp_image)

# ------------------
# Blur Faces
# ------------------
for detection in detection_result.detections:
    bbox = detection.bounding_box

    x1 = bbox.origin_x
    y1 = bbox.origin_y
    w = bbox.width
    h = bbox.height

    # Make sure coordinates are valid
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(iw, x1 + w)
    y2 = min(ih, y1 + h)

    face_region = img[y1:y2, x1:x2]
    # Moderate blur: smaller kernel and sigmaX
    blurred_face = cv2.GaussianBlur(face_region, (121, 121), 200)
    img[y1:y2, x1:x2] = blurred_face

# ------------------
# Show Result (Resize for display)
# ------------------
max_display_width = 800
max_display_height = 600
h, w = img.shape[:2]
scale = min(max_display_width / w, max_display_height / h, 1.0)
if scale < 1.0:
    display_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
else:
    display_img = img
cv2.imshow("Face Detection and Blurring", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

detector.close()