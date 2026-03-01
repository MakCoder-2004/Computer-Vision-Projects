# ---------------------------------
# Libraries
# ---------------------------------
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

# ---------------------------------
# TensorFlow GPU Optimization
# ---------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# Enable mixed precision (FP16 for RTX GPUs)
mixed_precision.set_global_policy('mixed_float16')

# Enable XLA acceleration
tf.config.optimizer.set_jit(True)

print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ---------------------------------
# Paths
# ---------------------------------
mask_path = 'data/model_data/mask.png'
video_path = 'data/inputs/parking_lot_video.mp4'
output_path = 'data/outputs/parking_lot_output.mp4'

# ---------------------------------
# Load parking spot coordinates
# ---------------------------------
def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []

    for i in range(1, totalLabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT])
        y1 = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])
        slots.append([x1, y1, w, h])

    return slots


mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_components)
spots_status = [None] * len(spots)
diffs = [0] * len(spots)

# ---------------------------------
# Load Model
# ---------------------------------
print("Loading AI model...")
MODEL = tf.keras.models.load_model('./Car Occurrence Model/model/car_occurrence_model.keras')
print("✓ Model loaded successfully\n")

# ---------------------------------
# Utility Functions
# ---------------------------------
def calculate_diff(im1, im2):
    return abs(np.mean(im1) - np.mean(im2))


def empty_or_not_batch(spot_bgr_list, batch_size=16):
    if len(spot_bgr_list) == 0:
        return []

    results = []

    for i in range(0, len(spot_bgr_list), batch_size):
        batch_spots = spot_bgr_list[i:i + batch_size]

        img_batch = []
        for spot_bgr in batch_spots:
            img_resized = cv2.resize(spot_bgr, (224, 224))
            img_resized = img_resized.astype(np.float16) / 255.0
            img_batch.append(img_resized)

        img_batch = np.array(img_batch, dtype=np.float16)

        y_output = MODEL(img_batch, training=False)
        y_output = y_output.numpy()

        results.extend([output[0] < 0.5 for output in y_output])

    return results

# ---------------------------------
# Load Video
# ---------------------------------
print("Loading video...")
video = cv2.VideoCapture(video_path)

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print(f"✓ Found {len(spots)} parking spots\n")

scale_factor = 0.75
display_width = int(frame_width * scale_factor)
display_height = int(frame_height * scale_factor)

# ---------------------------------
# Video Writer for Export
# ---------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (display_width, display_height))
# Set codec quality properties for better output
out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
print(f"✓ Video output will be saved to: {output_path}\n")

# ---------------------------------
# Processing
# ---------------------------------
print("Starting video processing...")
print("Press 'q' to quit\n")

step = 45
frame_count = 0
previous_frame = None
y_offset = -20  # Shift background and text upwards

while True:
    success, frame = video.read()
    if not success:
        break

    # ---------------------------------
    # Motion Detection
    # ---------------------------------
    if frame_count % step == 0 and previous_frame is not None:
        for i, (x1, y1, w, h) in enumerate(spots):
            current_crop = frame[y1:y1 + h, x1:x1 + w]
            previous_crop = previous_frame[y1:y1 + h, x1:x1 + w]
            diffs[i] = calculate_diff(current_crop, previous_crop)

    # ---------------------------------
    # AI Prediction
    # ---------------------------------
    if frame_count % step == 0:
        if previous_frame is None:
            indices = range(len(spots))
        else:
            max_diff = max(diffs) if max(diffs) != 0 else 1
            indices = [i for i, d in enumerate(diffs) if d / max_diff > 0.4]

        crops = []
        crop_indices = []

        for idx in indices:
            x1, y1, w, h = spots[idx]
            crop = frame[y1:y1 + h, x1:x1 + w]
            crops.append(crop)
            crop_indices.append(idx)

        if len(crops) > 0:
            statuses = empty_or_not_batch(crops)
            for i, idx in enumerate(crop_indices):
                spots_status[idx] = statuses[i]

        previous_frame = frame.copy()

    # ---------------------------------
    # Draw UI (dynamic background)
    # ---------------------------------
    available_spots = sum([1 for s in spots_status if s])

    text = f'Available Spots: {available_spots} / {len(spots_status)}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.2
    thickness = 4

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding_x = 20
    padding_y = 20

    # Draw rectangle with upward shift
    top_left = (50, 20 + y_offset)
    bottom_right = (50 + text_width + 2 * padding_x, 20 + text_height + 2 * padding_y + y_offset)
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)

    # Draw the text inside rectangle
    text_org = (50 + padding_x, 20 + text_height + padding_y - baseline + y_offset)
    cv2.putText(frame, text, text_org, font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    # Draw parking rectangles
    for i, (x1, y1, w, h) in enumerate(spots):
        status = spots_status[i]
        if status is not None:
            color = (0, 255, 0) if status else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

    # ---------------------------------
    # Resize & Display / Export
    # ---------------------------------
    display_frame = cv2.resize(frame, (display_width, display_height))
    cv2.imshow('Parking Slot Detection', display_frame)
    out.write(display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    frame_count += 1

video.release()
out.release()
cv2.destroyAllWindows()

print("\n✓ Video processing completed and saved!")