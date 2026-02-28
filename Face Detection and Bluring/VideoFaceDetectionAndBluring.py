import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize video capture
video = cv2.VideoCapture("data/inputs/test.mp4")

# Check if video opened successfully
if video is None or not video.isOpened():
    print("Error: Could not open the video file.")
    exit()

# Get video properties for export
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define VideoWriter to export processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/outputs/output_blurred.mp4', fourcc, fps, (width, height))

# Create Face Detector
base_options = python.BaseOptions(model_asset_path='model/detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# Process video frames
while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    ih, iw, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect faces in the frame
    detection_result = detector.detect(mp_image)

    for detection in detection_result.detections:
        bbox = detection.bounding_box

        x1 = int(bbox.origin_x)
        y1 = int(bbox.origin_y)
        w = int(bbox.width)
        h = int(bbox.height)

        # Ensure coordinates are within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(iw, x1 + w)
        y2 = min(ih, y1 + h)

        face_region = frame[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_region, (31,31), 50)
        frame[y1:y2, x1:x2] = blurred_face

    cv2.imshow('Face Detection and Blurring', frame)
    out.write(frame)  # Export processed frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

video.release()
out.release()  # Release VideoWriter
cv2.destroyAllWindows()
