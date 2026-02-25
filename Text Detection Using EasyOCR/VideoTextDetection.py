# ---------------------------------------
# Loading Libraries
# ---------------------------------------
import easyocr
import cv2

print(cv2.__version__)
print("Win32 UI support check:")
print("Win32 UI: YES" if "Win32 UI: YES" in cv2.getBuildInformation() else "NO")

# ---------------------------------------
# Loading Video
# ---------------------------------------
cap = cv2.VideoCapture(0)
reader = easyocr.Reader(['en'], gpu=False)
frame_count = 0
skip_frames = 10  # Process every 10th frame
cached_text = []  # Store results to show on skipped frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only run OCR every 'skip_frames' to improve speed
    if frame_count % skip_frames == 0:
        # Resize frame for faster OCR (reduce processing load)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        results = reader.readtext(small_frame)

        # Adjust coordinates back to original frame size
        cached_text = []
        for (bbox, text, confidence) in results:
            # Scale bbox coordinates back up (x2 because we resized by 0.5)
            new_bbox = [(int(pt[0] * 2), int(pt[1] * 2)) for pt in bbox]
            cached_text.append((new_bbox, text, confidence))

    # Displaying Cached Text Results
    for (bbox, text, confidence) in cached_text:
        top_left = tuple(bbox[0])
        bottom_right = tuple(bbox[2])

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.8
        thickness = 2

        # Draw background label
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        cv2.rectangle(frame, (top_left[0], top_left[1] - text_h - 10), (top_left[0] + text_w, top_left[1]), (0, 255, 0),
                      -1)

        cv2.putText(frame, text, (top_left[0], top_left[1] - 5), font, font_scale, (0, 0, 0), thickness)

    cv2.imshow('Video Text Detection (Optimized)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()