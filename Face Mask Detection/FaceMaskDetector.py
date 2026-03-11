import cv2
from ultralytics import YOLO

# ------------------------------------
# Load the trained YOLO model
# ------------------------------------
# Load the YOLO model
model = YOLO("models/best.onnx", task='detect')

# ------------------------------------
# Model Labels
# ------------------------------------
label_map = {
    0: "no_mask",      # mask_worn_incorrect → mask_worn_incorrect
    1: "mask",                     # with_mask → mask
}

# ------------------------------------
# Function to detect faces and masks
# ------------------------------------
def detect_faces(frame):
    results = model(frame)
    return results

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    if not check:
        print("Failed to capture video")
        break

    # Run detection
    results = detect_faces(frame)

    # Get the annotated frame
    annotated_frame = frame.copy()

    # ------------------------------------
    # Draw custom labels on the frame
    # ------------------------------------
    for box in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get class ID and confidence
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # Map to simplified label
        label = label_map.get(class_id, "unknown")

        # Choose color based on label (green=mask, red=no_mask, magenta=incorrect)
        if label == "mask":
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Draw label with confidence
        label_text = f"{label}: {confidence:.2f}"
        cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Face Mask Detector', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()